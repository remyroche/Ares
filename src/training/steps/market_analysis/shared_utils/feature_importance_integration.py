"""
Feature Importance Analysis Integration Module

This module provides seamless integration of feature importance analysis
into the broader market analysis pipeline, including clustering, reporting,
and evaluation components.

Key Integration Points:
- Pipeline Components: Hybrid orchestrator, regime discovery components
- Reporting: Unified evaluation framework, regime reporting
- Clustering: Pre/post clustering analysis
- Evaluation: Enhanced interpretability and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json

from .balanced_feature_extractor import (
    BalancedFeatureExtractor, BalancedFeatureConfig,
    analyze_regime_feature_importance
)

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportanceIntegrationConfig:
    """Configuration for feature importance integration."""
    # Analysis settings
    enable_pre_clustering_analysis: bool = True
    enable_post_clustering_analysis: bool = True
    enable_regime_characterization: bool = True
    enable_feature_validation: bool = True

    # Methods and thresholds
    importance_methods: List[str] = field(default_factory=lambda: [
        "mutual_information", "f_classif", "variance"
    ])
    importance_threshold: float = 0.01  # Minimum importance threshold
    min_features_for_analysis: int = 10

    # Reporting settings
    include_detailed_analysis: bool = True
    save_feature_profiles: bool = True
    generate_interpretation: bool = True

    # Integration settings
    auto_integrate_with_clustering: bool = True
    auto_integrate_with_reporting: bool = True

class FeatureImportanceIntegrationManager:
    """
    Manages feature importance analysis integration across the pipeline.

    This manager coordinates feature importance analysis at multiple pipeline stages:
    1. Pre-clustering: Feature validation and selection
    2. Post-clustering: Regime characterization and validation
    3. Reporting: Enhanced interpretability and insights
    """

    def __init__(self, config: Optional[FeatureImportanceIntegrationConfig] = None):
        """Initialize the feature importance integration manager."""
        self.config = config or FeatureImportanceIntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Analysis results storage
        self.pre_clustering_analysis = {}
        self.post_clustering_analysis = {}
        self.regime_characterizations = {}

    def analyze_pre_clustering_features(self, features: np.ndarray,
                                      feature_names: List[str],
                                      labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze feature importance before clustering to guide feature selection.

        This analysis helps:
        - Identify most discriminative features for clustering
        - Validate feature quality and relevance
        - Guide feature selection and dimensionality reduction
        """
        self.logger.info("🔍 Analyzing pre-clustering feature importance")

        try:
            analysis_results = {}

            for method in self.config.importance_methods:
                if features.shape[1] < self.config.min_features_for_analysis:
                    self.logger.warning(f"⚠️ Insufficient features ({features.shape[1]}) for {method} analysis")
                    continue

                # Calculate feature importance using the balanced feature extractor
                importance_result = analyze_regime_feature_importance(
                    features=features,
                    feature_names=feature_names,
                    regime_labels=labels if labels is not None else np.zeros(features.shape[0]),
                    method=method,
                    config=BalancedFeatureConfig()
                )

                if importance_result:
                    analysis_results[method] = importance_result

                    # Extract key insights
                    key_features = importance_result.get('most_important_features', [])
                    interpretation = importance_result.get('interpretation', '')

                    self.logger.info(f"✅ {method} analysis completed. Key features: {key_features[:5]}")
                    self.logger.info(f"📊 Interpretation: {interpretation}")

            self.pre_clustering_analysis = analysis_results
            return analysis_results

        except Exception as e:
            self.logger.error(f"❌ Pre-clustering feature analysis failed: {e}")
            return {}

    def analyze_post_clustering_regimes(self, features: np.ndarray,
                                     feature_names: List[str],
                                     regime_labels: np.ndarray,
                                     clusterer_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze feature importance after clustering for regime characterization.

        This analysis provides:
        - Regime-specific feature profiles
        - Most discriminative features per regime
        - Regime interpretability and validation
        - Clustering quality assessment
        """
        self.logger.info("🔍 Analyzing post-clustering regime feature importance")

        try:
            analysis_results = {}

            # Validate inputs
            unique_regimes = len(np.unique(regime_labels))
            if unique_regimes < 2:
                self.logger.warning(f"⚠️ Only {unique_regimes} regimes found - may limit analysis quality")
                return {}

            for method in self.config.importance_methods:
                if features.shape[1] < self.config.min_features_for_analysis:
                    continue

                # Perform comprehensive regime analysis
                importance_result = analyze_regime_feature_importance(
                    features=features,
                    feature_names=feature_names,
                    regime_labels=regime_labels,
                    method=method,
                    config=BalancedFeatureConfig()
                )

                if importance_result:
                    analysis_results[method] = importance_result

                    # Store regime characterizations
                    regime_profiles = importance_result.get('regime_feature_profiles', {})
                    for regime_id, profile in regime_profiles.items():
                        self.regime_characterizations[regime_id] = {
                            'method': method,
                            'profile': profile,
                            'dominant_features': profile.get('dominant_features', []),
                            'timestamp': datetime.now()
                        }

                    # Log key insights
                    interpretation = importance_result.get('interpretation', '')
                    separability = importance_result.get('regime_separability', {})

                    self.logger.info(f"✅ {method} regime analysis completed")
                    self.logger.info(f"📊 Regime separability: {separability}")
                    self.logger.info(f"🔍 Interpretation: {interpretation}")

            self.post_clustering_analysis = analysis_results
            return analysis_results

        except Exception as e:
            self.logger.error(f"❌ Post-clustering regime analysis failed: {e}")
            return {}

    def generate_enhanced_regime_report(self, base_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing regime reports with feature importance analysis.

        Args:
            base_report: Existing regime report structure

        Returns:
            Enhanced report with feature importance insights
        """
        try:
            enhanced_report = base_report.copy()

            # Add feature importance section
            feature_importance_section = {
                'pre_clustering_analysis': self.pre_clustering_analysis,
                'post_clustering_analysis': self.post_clustering_analysis,
                'regime_characterizations': self.regime_characterizations,
                'integration_config': {
                    'importance_methods': self.config.importance_methods,
                    'importance_threshold': self.config.importance_threshold,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }

            # Merge with existing report structure
            if 'analysis' not in enhanced_report:
                enhanced_report['analysis'] = {}

            enhanced_report['analysis']['feature_importance'] = feature_importance_section

            # Add summary insights
            if self.regime_characterizations:
                summary_insights = self._generate_summary_insights()
                enhanced_report['summary_insights'] = summary_insights

            self.logger.info("✅ Enhanced regime report generated with feature importance analysis")
            return enhanced_report

        except Exception as e:
            self.logger.error(f"❌ Failed to generate enhanced regime report: {e}")
            return base_report

    def _generate_summary_insights(self) -> Dict[str, Any]:
        """Generate summary insights from feature importance analysis."""
        try:
            insights = {
                'total_regime_analyses': len(self.regime_characterizations),
                'feature_coverage': {},
                'regime_stability_indicators': {},
                'key_discriminators': set()
            }

            # Aggregate feature importance across regimes
            feature_importance_scores = {}
            regime_feature_counts = {}

            for regime_id, characterization in self.regime_characterizations.items():
                dominant_features = characterization.get('dominant_features', [])

                for feature in dominant_features:
                    if feature not in feature_importance_scores:
                        feature_importance_scores[feature] = 0
                        regime_feature_counts[feature] = 0

                    feature_importance_scores[feature] += 1
                    regime_feature_counts[feature] += 1

                insights['key_discriminators'].update(dominant_features)

            # Calculate feature coverage and stability
            total_regimes = len(self.regime_characterizations)
            insights['feature_coverage'] = {
                feature: count / total_regimes
                for feature, count in feature_importance_scores.items()
            }

            insights['regime_stability_indicators'] = {
                'avg_regimes_per_feature': np.mean(list(regime_feature_counts.values())),
                'max_regimes_per_feature': max(regime_feature_counts.values()),
                'unique_discriminators': len(insights['key_discriminators'])
            }

            return insights

        except Exception as e:
            self.logger.warning(f"Failed to generate summary insights: {e}")
            return {}

def integrate_feature_importance_with_clustering(
    clusterer,
    features: np.ndarray,
    feature_names: List[str],
    integration_config: Optional[FeatureImportanceIntegrationConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to integrate feature importance analysis with clustering.

    This function automatically:
    1. Analyzes features before clustering (if enabled)
    2. Performs clustering
    3. Analyzes regime characteristics after clustering
    4. Returns enhanced clustering results

    Args:
        clusterer: Clustering algorithm instance
        features: Feature matrix
        feature_names: Feature names
        integration_config: Integration configuration

    Returns:
        Enhanced clustering results with feature importance analysis
    """
    integration_manager = FeatureImportanceIntegrationManager(integration_config)

    try:
        # Pre-clustering analysis
        if integration_config and integration_config.enable_pre_clustering_analysis:
            logger.info("🔍 Performing pre-clustering feature importance analysis")
            integration_manager.analyze_pre_clustering_features(features, feature_names)

        # Perform clustering
        logger.info("🔄 Performing clustering")
        cluster_labels = clusterer.fit_predict(features)

        # Post-clustering analysis
        if integration_config and integration_config.enable_post_clustering_analysis:
            logger.info("🔍 Performing post-clustering regime feature importance analysis")
            integration_manager.analyze_post_clustering_regimes(
                features, feature_names, cluster_labels,
                {'clusterer_type': type(clusterer).__name__}
            )

        # Return enhanced results
        enhanced_results = {
            'cluster_labels': cluster_labels,
            'feature_importance_analysis': {
                'pre_clustering': integration_manager.pre_clustering_analysis,
                'post_clustering': integration_manager.post_clustering_analysis,
                'regime_characterizations': integration_manager.regime_characterizations
            },
            'clustering_info': {
                'n_clusters': len(np.unique(cluster_labels)),
                'clusterer_type': type(clusterer).__name__,
                'n_samples': features.shape[0],
                'n_features': features.shape[1]
            }
        }

        return enhanced_results

    except Exception as e:
        logger.error(f"❌ Feature importance integration with clustering failed: {e}")
        # Return basic clustering results as fallback
        return {
            'cluster_labels': clusterer.fit_predict(features) if hasattr(clusterer, 'fit_predict') else np.zeros(features.shape[0]),
            'feature_importance_analysis': {},
            'clustering_info': {'error': str(e)}
        }

def enhance_regime_report_with_feature_importance(
    base_report: Dict[str, Any],
    feature_importance_manager: FeatureImportanceIntegrationManager
) -> Dict[str, Any]:
    """
    Enhance an existing regime report with feature importance analysis.

    Args:
        base_report: Existing regime report
        feature_importance_manager: Configured integration manager

    Returns:
        Enhanced report with feature importance insights
    """
    return feature_importance_manager.generate_enhanced_regime_report(base_report)

# Integration hooks for pipeline components
class FeatureImportancePipelineHook:
    """
    Pipeline hook for automatic feature importance integration.

    This hook can be attached to pipeline components to automatically
    inject feature importance analysis at appropriate stages.
    """

    def __init__(self, integration_config: Optional[FeatureImportanceIntegrationConfig] = None):
        self.integration_manager = FeatureImportanceIntegrationManager(integration_config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def pre_clustering_hook(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Hook called before clustering."""
        if self.integration_manager.config.enable_pre_clustering_analysis:
            return self.integration_manager.analyze_pre_clustering_features(features, feature_names)
        return {}

    def post_clustering_hook(self, features: np.ndarray, feature_names: List[str],
                           cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Hook called after clustering."""
        if self.integration_manager.config.enable_post_clustering_analysis:
            return self.integration_manager.analyze_post_clustering_regimes(
                features, feature_names, cluster_labels
            )
        return {}

    def reporting_hook(self, base_report: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during reporting."""
        if self.integration_manager.config.auto_integrate_with_reporting:
            return self.integration_manager.generate_enhanced_regime_report(base_report)
        return base_report
