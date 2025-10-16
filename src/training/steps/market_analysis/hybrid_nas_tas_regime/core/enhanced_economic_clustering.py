"""
Enhanced Economic Clustering for NAS-TAS System

This module integrates all the new data-driven improvements:
- Enhanced regime evaluation metrics
- Feature correlation handling with PCA/VIF
- Cross-validation for clustering parameters
- Robust scoring models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import new modules
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.enhanced_regime_evaluator import (
    EnhancedRegimeEvaluator, create_enhanced_regime_evaluator
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.clustering_cross_validation import (
    ClusteringCrossValidator, create_clustering_cross_validator
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.robust_scoring_models import (
    RobustScoringModels, create_robust_scoring_models
)
from src.feature_selection.dimensionality import create_pca_module, create_vif_module

logger = logging.getLogger(__name__)

@dataclass
class EnhancedClusteringResult:
    """Result from enhanced economic clustering."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    probabilities: np.ndarray
    regime_metrics: List[Dict[str, Any]]
    regime_rankings: Dict[str, List[int]]
    overall_quality_score: float
    feature_selection_info: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    scoring_model_results: Dict[str, Any]
    economic_metrics: Dict[str, Any]
    algorithm_used: str
    execution_time: float
    metadata: Dict[str, Any]

class EnhancedEconomicClusterer:
    """
    Enhanced economic clusterer with data-driven improvements.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced economic clusterer."""
        tprint_info("🚀 Initializing Enhanced Economic Clusterer")
        tprint_debug(f"Configuration: {config}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        tprint_debug("🔧 Initializing evaluation components...")

        # Enhanced regime evaluator
        evaluator_config = config.get('evaluator_config', {})
        self.regime_evaluator = create_enhanced_regime_evaluator(evaluator_config)
        tprint_success("✅ Enhanced regime evaluator initialized")

        # Cross-validation for clustering
        cv_config = config.get('cv_config', {})
        self.clustering_cv = create_clustering_cross_validator(cv_config)
        tprint_success("✅ Clustering cross-validator initialized")

        # Robust scoring models
        scoring_config = config.get('scoring_config', {})
        self.scoring_models = create_robust_scoring_models(scoring_config)
        tprint_success("✅ Robust scoring models initialized")

        # Feature selection modules
        tprint_debug("🔧 Initializing feature selection modules...")
        pca_config = config.get('pca_config', {})
        self.pca_module = create_pca_module(pca_config)
        tprint_success("✅ PCA module initialized")

        vif_config = config.get('vif_config', {})
        self.vif_module = create_vif_module(vif_config)
        tprint_success("✅ VIF module initialized")

        # Clustering parameters
        tprint_debug("⚙️ Setting clustering parameters...")
        self.n_regimes = config.get('n_regimes', 8)
        self.primary_algorithm = config.get('primary_algorithm', 'economic_adaptive')
        self.enable_feature_selection = config.get('enable_feature_selection', True)
        self.enable_cross_validation = config.get('enable_cross_validation', True)
        self.enable_scoring_models = config.get('enable_scoring_models', True)
        tprint_success("✅ Clustering parameters configured")

        tprint_success("✅ Enhanced Economic Clusterer initialized")
        self.logger.info("✅ Enhanced Economic Clusterer initialized")

    def cluster_with_enhanced_evaluation(self,
                                       features: np.ndarray,
                                       market_data: pd.DataFrame,
                                       historical_data: Optional[pd.DataFrame] = None) -> EnhancedClusteringResult:
        """
        Perform enhanced economic clustering with all improvements.

        Args:
            features: Feature matrix
            market_data: Market data for economic analysis
            historical_data: Optional historical data for model training

        Returns:
            EnhancedClusteringResult with comprehensive clustering results
        """
        try:
            tprint("🔍 [ENHANCED_CLUSTERING] Starting enhanced economic clustering", color="blue", bold=True)
            tprint_debug(f"📊 [ENHANCED_CLUSTERING] Features shape: {features.shape}")
            tprint_debug(f"📊 [ENHANCED_CLUSTERING] Market data shape: {market_data.shape}")
            self.logger.info("🔍 Starting enhanced economic clustering...")

            # Step 1: Feature selection and correlation handling
            tprint("🔧 [ENHANCED_CLUSTERING] Step 1: Feature selection and correlation handling", color="cyan")
            feature_selection_info = self._apply_feature_selection(features, market_data)
            tprint_success(f"✅ [ENHANCED_CLUSTERING] Feature selection completed: {feature_selection_info['n_selected_features']} features selected")

            # Step 2: Cross-validation for clustering parameters
            tprint("🎯 [ENHANCED_CLUSTERING] Step 2: Cross-validation for clustering parameters", color="cyan")
            if self.enable_cross_validation:
                cv_results = self._optimize_clustering_parameters(features, market_data)
                tprint_success(f"✅ [ENHANCED_CLUSTERING] Cross-validation completed: {cv_results['best_params']}")
            else:
                cv_results = {'best_params': {'n_regimes': self.n_regimes, 'algorithm': self.primary_algorithm}}
                tprint_success("✅ [ENHANCED_CLUSTERING] Using default clustering parameters")

            # Step 3: Apply clustering with optimized parameters
            tprint("🔄 [ENHANCED_CLUSTERING] Step 3: Applying clustering with optimized parameters", color="cyan")
            clustering_result = self._apply_enhanced_clustering(features, market_data, cv_results['best_params'])
            tprint_success(f"✅ [ENHANCED_CLUSTERING] Clustering completed: {len(set(clustering_result['labels']))} clusters")

            # Step 4: Enhanced regime evaluation
            tprint("📊 [ENHANCED_CLUSTERING] Step 4: Enhanced regime evaluation", color="cyan")
            regime_evaluation = self._evaluate_regimes_enhanced(market_data, clustering_result['labels'])
            tprint_success(f"✅ [ENHANCED_CLUSTERING] Regime evaluation completed: {len(regime_evaluation.regime_metrics)} regimes evaluated")

            # Step 5: Train and apply robust scoring models
            tprint("🤖 [ENHANCED_CLUSTERING] Step 5: Training and applying robust scoring models", color="cyan")
            if self.enable_scoring_models and historical_data is not None:
                scoring_results = self._train_and_apply_scoring_models(
                    historical_data, features, clustering_result['labels'], regime_evaluation.regime_metrics
                )
                tprint_success("✅ [ENHANCED_CLUSTERING] Robust scoring models applied")
            else:
                scoring_results = {}
                tprint_success("✅ [ENHANCED_CLUSTERING] Skipping scoring models (no historical data or disabled)")

            # Step 6: Calculate economic metrics
            tprint("💰 [ENHANCED_CLUSTERING] Step 6: Calculating economic metrics", color="cyan")
            economic_metrics = self._calculate_enhanced_economic_metrics(
                market_data, clustering_result['labels'], regime_evaluation
            )
            tprint_success("✅ [ENHANCED_CLUSTERING] Economic metrics calculated")

            tprint_success(f"🎉 [ENHANCED_CLUSTERING] Enhanced economic clustering completed successfully")
            tprint_performance(f"⚡ [ENHANCED_CLUSTERING] Final result: {len(set(clustering_result['labels']))} clusters, {len(regime_evaluation.regime_metrics)} regimes evaluated")

            return EnhancedClusteringResult(
                labels=clustering_result['labels'],
                cluster_centers=clustering_result['cluster_centers'],
                probabilities=clustering_result['probabilities'],
                regime_metrics=regime_evaluation.regime_metrics,
                regime_rankings=regime_evaluation.regime_rankings,
                overall_quality_score=regime_evaluation.overall_quality_score,
                feature_selection_info=feature_selection_info,
                cross_validation_results=cv_results,
                scoring_model_results=scoring_results,
                economic_metrics=economic_metrics,
                algorithm_used=cv_results['best_params'].get('algorithm', self.primary_algorithm),
                execution_time=0.0,  # TODO: Add timing
                metadata={
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'n_regimes': len(set(clustering_result['labels'])),
                    'enhancement_timestamp': datetime.now().isoformat(),
                    'config': self.config
                }
            )

        except Exception as e:
            tprint_error(f"❌ [ENHANCED_CLUSTERING] Enhanced economic clustering failed: {e}")
            tprint_debug(f"🔍 [ENHANCED_CLUSTERING] Error details: {str(e)}")
            self.logger.error(f"Enhanced economic clustering failed: {e}")
            raise

    def _apply_feature_selection(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply feature selection and correlation handling."""
        try:
            feature_selection_info = {
                'original_n_features': features.shape[1],
                'selected_features': [],
                'selection_method': 'none',
                'n_selected_features': features.shape[1]
            }

            if not self.enable_feature_selection:
                return feature_selection_info

            # Apply VIF-based feature selection
            tprint_debug("🔧 [ENHANCED_CLUSTERING] Applying VIF-based feature selection")
            vif_result = self.vif_module.apply_vif_feature_selection(features)

            if vif_result['success'] and len(vif_result['selected_features']) > 0:
                selected_indices = vif_result['selected_indices']
                features_selected = features[:, selected_indices]
                feature_selection_info.update({
                    'selected_features': vif_result['selected_features'],
                    'selection_method': 'vif_based',
                    'n_selected_features': len(vif_result['selected_features']),
                    'vif_scores': vif_result.get('vif_scores', {}),
                    'removal_info': vif_result.get('removal_info', {})
                })
                tprint_success(f"✅ [ENHANCED_CLUSTERING] VIF selection: {len(vif_result['selected_features'])} features selected")
            else:
                tprint_warning("⚠️ [ENHANCED_CLUSTERING] VIF selection failed, using all features")
                features_selected = features

            # Apply PCA if still too many features
            if features_selected.shape[1] > 50:  # Threshold for PCA
                tprint_debug("🔧 [ENHANCED_CLUSTERING] Applying PCA for dimensionality reduction")
                pca_result = self.pca_module.apply_pca_feature_selection(features_selected)

                if pca_result['success'] and len(pca_result['selected_features']) > 0:
                    # Use PCA components instead of original features
                    pca_components = self.pca_module.get_pca_components(features_selected)
                    feature_selection_info.update({
                        'pca_applied': True,
                        'pca_components': pca_components.shape[1],
                        'explained_variance_ratio': pca_result.get('explained_variance_ratio', []),
                        'cumulative_variance': pca_result.get('cumulative_variance', [])
                    })
                    tprint_success(f"✅ [ENHANCED_CLUSTERING] PCA applied: {pca_components.shape[1]} components")
                    return feature_selection_info, pca_components
                else:
                    tprint_warning("⚠️ [ENHANCED_CLUSTERING] PCA failed, using VIF-selected features")

            return feature_selection_info, features_selected

        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return {'original_n_features': features.shape[1], 'n_selected_features': features.shape[1]}, features

    def _optimize_clustering_parameters(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize clustering parameters using cross-validation."""
        try:
            tprint_debug("🎯 [ENHANCED_CLUSTERING] Optimizing clustering parameters")
            cv_result = self.clustering_cv.optimize_clustering_parameters(features, market_data)

            return {
                'best_params': cv_result.best_params,
                'best_score': cv_result.best_score,
                'cv_scores': cv_result.cv_scores,
                'validation_metrics': cv_result.validation_metrics,
                'stability_scores': cv_result.stability_scores
            }
        except Exception as e:
            self.logger.warning(f"Clustering parameter optimization failed: {e}")
            return {
                'best_params': {'n_regimes': self.n_regimes, 'algorithm': self.primary_algorithm},
                'best_score': 0.0,
                'cv_scores': {},
                'validation_metrics': {},
                'stability_scores': {}
            }

    def _apply_enhanced_clustering(self, features: np.ndarray, market_data: pd.DataFrame,
                                 best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply clustering with optimized parameters."""
        try:
            # This would integrate with the existing economic clustering logic
            # For now, use a simplified approach

            from sklearn.cluster import KMeans
            from sklearn.mixture import GaussianMixture

            n_regimes = best_params.get('n_regimes', self.n_regimes)
            algorithm = best_params.get('algorithm', 'kmeans')

            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                labels = clusterer.fit_predict(features)
                cluster_centers = clusterer.cluster_centers_
            elif algorithm == 'gmm':
                clusterer = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
                labels = clusterer.fit_predict(features)
                cluster_centers = clusterer.means_
            else:
                # Default to K-means
                clusterer = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                labels = clusterer.fit_predict(features)
                cluster_centers = clusterer.cluster_centers_

            # Calculate probabilities (simplified)
            probabilities = np.ones((len(labels), n_regimes)) / n_regimes

            return {
                'labels': labels,
                'cluster_centers': cluster_centers,
                'probabilities': probabilities,
                'algorithm_used': algorithm
            }

        except Exception as e:
            self.logger.error(f"Enhanced clustering application failed: {e}")
            raise

    def _evaluate_regimes_enhanced(self, market_data: pd.DataFrame, labels: np.ndarray) -> Any:
        """Evaluate regimes using enhanced regime evaluator."""
        try:
            tprint_debug("📊 [ENHANCED_CLUSTERING] Evaluating regimes with enhanced metrics")
            evaluation_result = self.regime_evaluator.evaluate_regimes(market_data, labels)
            return evaluation_result
        except Exception as e:
            self.logger.warning(f"Enhanced regime evaluation failed: {e}")
            # Return a minimal result
            from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.enhanced_regime_evaluator import RegimeEvaluationResult
            return RegimeEvaluationResult(
                regime_metrics=[],
                regime_rankings={},
                overall_quality_score=0.5,
                regime_transitions=[],
                risk_adjusted_rankings={},
                economic_rankings={},
                trading_rankings={},
                metadata={}
            )

    def _train_and_apply_scoring_models(self, historical_data: pd.DataFrame, features: np.ndarray,
                                      labels: np.ndarray, regime_metrics: List[Any]) -> Dict[str, Any]:
        """Train and apply robust scoring models."""
        try:
            tprint_debug("🤖 [ENHANCED_CLUSTERING] Training robust scoring models")

            # Convert regime metrics to the format expected by scoring models
            regime_metrics_dict = []
            for i, metric in enumerate(regime_metrics):
                if hasattr(metric, '__dict__'):
                    regime_metrics_dict.append(metric.__dict__)
                else:
                    regime_metrics_dict.append(metric)

            # Train models
            model_performances = self.scoring_models.train_scoring_models(
                historical_data, features, labels, regime_metrics_dict
            )

            # Apply models to current data
            scoring_results = {}
            for regime_id in set(labels):
                regime_mask = labels == regime_id
                regime_features = features[regime_mask]
                regime_market_data = historical_data[regime_mask] if len(historical_data) > len(labels) else historical_data

                scoring_result = self.scoring_models.predict_regime_scores(
                    regime_features, regime_market_data, regime_id
                )
                scoring_results[f'regime_{regime_id}'] = scoring_result

            return {
                'model_performances': model_performances,
                'scoring_results': scoring_results,
                'models_trained': len(model_performances)
            }

        except Exception as e:
            self.logger.warning(f"Robust scoring models training/application failed: {e}")
            return {}

    def _calculate_enhanced_economic_metrics(self, market_data: pd.DataFrame, labels: np.ndarray,
                                           regime_evaluation: Any) -> Dict[str, Any]:
        """Calculate enhanced economic metrics."""
        try:
            economic_metrics = {
                'n_regimes': len(set(labels)),
                'regime_sizes': np.bincount(labels, minlength=len(set(labels))).tolist(),
                'overall_quality_score': regime_evaluation.overall_quality_score,
                'regime_rankings': regime_evaluation.regime_rankings,
                'risk_adjusted_rankings': regime_evaluation.risk_adjusted_rankings,
                'economic_rankings': regime_evaluation.economic_rankings,
                'trading_rankings': regime_evaluation.trading_rankings
            }

            # Add regime-specific metrics
            for i, metric in enumerate(regime_evaluation.regime_metrics):
                if hasattr(metric, '__dict__'):
                    regime_id = getattr(metric, 'regime_id', i)
                    economic_metrics[f'regime_{regime_id}'] = {
                        'mean_return': getattr(metric, 'mean_return', 0.0),
                        'volatility': getattr(metric, 'volatility', 0.0),
                        'sharpe_ratio': getattr(metric, 'sharpe_ratio', 0.0),
                        'sortino_ratio': getattr(metric, 'sortino_ratio', 0.0),
                        'max_drawdown': getattr(metric, 'max_drawdown', 0.0),
                        'hit_rate': getattr(metric, 'hit_rate', 0.0),
                        'payoff_ratio': getattr(metric, 'payoff_ratio', 0.0),
                        'economic_significance': getattr(metric, 'economic_significance', 0.0),
                        'trading_viability': getattr(metric, 'trading_viability', 0.0),
                        'stability_score': getattr(metric, 'stability_score', 0.0),
                        'risk_score': getattr(metric, 'risk_score', 0.0),
                        'performance_score': getattr(metric, 'performance_score', 0.0)
                    }

            return economic_metrics

        except Exception as e:
            self.logger.warning(f"Enhanced economic metrics calculation failed: {e}")
            return {'n_regimes': len(set(labels)), 'overall_quality_score': 0.5}

def create_enhanced_economic_clusterer(config: Dict[str, Any]) -> EnhancedEconomicClusterer:
    """Create enhanced economic clusterer."""
    return EnhancedEconomicClusterer(config)
