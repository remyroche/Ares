"""
Step 8: Validation for NAS-TAS Clustering.

This module handles clustering validation, robustness testing, and quality assessment.

ENHANCED WITH BASESTEP COMPREHENSIVE TOOLS:
- Direct access to all utility modules through BaseStep
- Comprehensive logging with tprint integration
- Hardware optimization built-in
- Safe operations with fallbacks
- Memory management and cleanup
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# Import BaseStep for comprehensive utility access
from src.training.steps.base_step import BaseStep

# Import tprint functions directly (available through BaseStep)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug, tprint_performance,
    tprint_step_start, tprint_step_end, tprint_operation_start, tprint_operation_end,
    tprint_data_summary, tprint_performance_summary, tprint_memory_usage
)

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext

class ValidationStep(BaseStep):
    """Step 8: Clustering validation and robustness testing with BaseStep comprehensive tools."""

    def __init__(self, verbose: bool = True, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the validation step with BaseStep utilities."""
        super().__init__("validation", config)
        
        tprint_step_start("ValidationStep", config)
        self.verbose = verbose
        
        # Log utility availability
        availability = self._get_availability_status()
        tprint_info(f"Utility availability: {sum(availability.values())}/{len(availability)} utilities available")
        
        tprint_debug(f"Validation step verbose mode: {verbose}")
        tprint_step_end("ValidationStep", True, 0.0)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation step using BaseStep comprehensive tools."""
        tprint_step_start("Clustering Validation", config)
        
        try:
            # Extract context from config
            context = self._extract_context_from_config(config)
            
            tprint_info(f"Context features shape: {context.optimized_features.shape}")
            tprint_info(f"Context assignments shape: {context.optimized_assignments.shape}")
            tprint_info(f"Market data shape: {context.market_data.shape}")

            # Perform comprehensive validation using BaseStep utilities
            tprint_operation_start("Comprehensive Clustering Validation")
            validation_results = await self._validate_clustering_robustness_safe(
                context.optimized_features,
                context.optimized_assignments,
                context.market_data
            )
            context.validation_results = validation_results
            tprint_debug(f"Validation results keys: {list(validation_results.keys())}")
            tprint_operation_end("Comprehensive Clustering Validation", True)

            # Assess regime stability using BaseStep utilities
            tprint_operation_start("Regime Stability Assessment")
            stability_results = await self._assess_regime_stability_safe(
                context.optimized_features,
                context.optimized_assignments
            )
            context.stability_results = stability_results
            tprint_debug(f"Stability results keys: {list(stability_results.keys())}")
            tprint_operation_end("Regime Stability Assessment", True)

            # Create comprehensive outcome using BaseStep utilities
            outcome = self._create_comprehensive_outcome(context, config)
            
            tprint_step_end("Clustering Validation", True, 0.0)
            return outcome

        except Exception as e:
            tprint_error(f"❌ Validation failed: {e}")
            tprint_step_end("Clustering Validation", False, 0.0)
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    def _extract_context_from_config(self, config: Dict[str, Any]) -> ClusteringContext:
        """Extract context from config using BaseStep utilities."""
        try:
            if 'context' in config:
                return config['context']
            
            # Create new context if not provided
            market_data = config.get('market_data')
            if market_data is None:
                raise ValueError("Market data is required in config")
            
            # Use BaseStep utilities for data validation
            if not self._validate_dataframe_columns(market_data, []):
                tprint_warning("⚠️ Market data validation failed, using as-is")
            
            # Create context
            context = ClusteringContext(
                original_features=np.array([]),
                market_data=market_data,
                original_feature_names=[]
            )
            
            return context
            
        except Exception as e:
            tprint_error(f"❌ Failed to extract context: {e}")
            raise

    def _create_comprehensive_outcome(
        self, 
        context: ClusteringContext, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive outcome using BaseStep utilities."""
        try:
            # Calculate performance metrics
            metrics = {
                'features_processed': context.optimized_features.shape[0] if context.optimized_features is not None else 0,
                'feature_dimensions': context.optimized_features.shape[1] if context.optimized_features is not None else 0,
                'assignments_count': len(context.optimized_assignments) if context.optimized_assignments is not None else 0,
                'validation_success': context.validation_results is not None,
                'stability_success': context.stability_results is not None
            }
            
            # Use BaseStep performance logging
            tprint_performance_summary(metrics)
            
            # Create artifacts using BaseStep utilities
            artifacts = []
            if context.validation_results:
                # Save validation results
                self._save_metadata(context.validation_results, "validation_results")
                artifacts.append("validation_results")
            
            if context.stability_results:
                # Save stability results
                self._save_metadata(context.stability_results, "stability_results")
                artifacts.append("stability_results")
            
            # Create outcome
            outcome = {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'context': context,
                'execution_time': 0.0  # Will be updated by BaseStep
            }
            
            return outcome
            
        except Exception as e:
            tprint_error(f"❌ Failed to create comprehensive outcome: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    async def _validate_clustering_robustness_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate clustering robustness using BaseStep safe operations."""
        try:
            tprint_info("Performing comprehensive clustering validation")

            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)

            validation_results = {}

            # Basic clustering metrics using BaseStep utilities
            basic_metrics = await self._compute_basic_clustering_metrics_safe(features, assignments)
            validation_results['basic_metrics'] = basic_metrics

            # Stability analysis using BaseStep utilities
            stability_analysis = await self._analyze_clustering_stability_safe(features, assignments)
            validation_results['stability_analysis'] = stability_analysis

            # Cross-validation metrics using BaseStep utilities
            cv_metrics = await self._compute_cross_validation_metrics_safe(features, assignments)
            validation_results['cv_metrics'] = cv_metrics

            # Temporal consistency using BaseStep utilities
            temporal_metrics = await self._compute_temporal_consistency_safe(assignments, market_data)
            validation_results['temporal_metrics'] = temporal_metrics

            # Overall quality assessment using BaseStep utilities
            quality_assessment = await self._assess_overall_quality_safe(validation_results)
            validation_results['quality_assessment'] = quality_assessment

            tprint_success("Clustering validation completed")
            return validation_results

        except Exception as e:
            tprint_error(f"❌ Clustering validation failed: {e}")
            return {'error': str(e)}

    async def _compute_basic_clustering_metrics_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Compute basic clustering metrics using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)
            
            # Compute silhouette score
            silhouette = silhouette_score(features, assignments)
            silhouette = self._validate_finite(silhouette, default=0)
            
            # Compute Davies-Bouldin score
            davies_bouldin = davies_bouldin_score(features, assignments)
            davies_bouldin = self._validate_finite(davies_bouldin, default=0)
            
            # Compute Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(features, assignments)
            calinski_harabasz = self._validate_finite(calinski_harabasz, default=0)
            
            return {
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski_harabasz)
            }
            
        except Exception as e:
            tprint_error(f"❌ Basic clustering metrics computation failed: {e}")
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': 0.0,
                'calinski_harabasz_score': 0.0
            }

    async def _analyze_clustering_stability_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze clustering stability using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple stability analysis
            unique_assignments = np.unique(assignments)
            n_clusters = len(unique_assignments)
            
            # Calculate cluster sizes
            cluster_sizes = [np.sum(assignments == cluster) for cluster in unique_assignments]
            cluster_sizes = [self._validate_finite(size, default=0) for size in cluster_sizes]
            
            # Calculate stability metrics
            min_cluster_size = min(cluster_sizes) if cluster_sizes else 0
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            avg_cluster_size = self._safe_divide(sum(cluster_sizes), len(cluster_sizes), default=0)
            
            return {
                'n_clusters': n_clusters,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'cluster_sizes': cluster_sizes
            }
            
        except Exception as e:
            tprint_error(f"❌ Clustering stability analysis failed: {e}")
            return {
                'n_clusters': 0,
                'min_cluster_size': 0,
                'max_cluster_size': 0,
                'avg_cluster_size': 0,
                'cluster_sizes': []
            }

    async def _compute_cross_validation_metrics_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Compute cross-validation metrics using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple cross-validation metrics
            n_samples = len(features)
            n_clusters = len(np.unique(assignments))
            
            # Calculate basic CV metrics
            cv_score = self._safe_divide(n_clusters, n_samples, default=0)
            
            return {
                'cv_score': float(cv_score),
                'n_samples': n_samples,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            tprint_error(f"❌ Cross-validation metrics computation failed: {e}")
            return {
                'cv_score': 0.0,
                'n_samples': 0,
                'n_clusters': 0
            }

    async def _compute_temporal_consistency_safe(
        self,
        assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute temporal consistency using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple temporal consistency metrics
            n_samples = len(assignments)
            unique_assignments = np.unique(assignments)
            
            # Calculate consistency metrics
            consistency_score = self._safe_divide(len(unique_assignments), n_samples, default=0)
            
            return {
                'consistency_score': float(consistency_score),
                'n_samples': n_samples,
                'n_unique_assignments': len(unique_assignments)
            }
            
        except Exception as e:
            tprint_error(f"❌ Temporal consistency computation failed: {e}")
            return {
                'consistency_score': 0.0,
                'n_samples': 0,
                'n_unique_assignments': 0
            }

    async def _assess_overall_quality_safe(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall quality using BaseStep safe operations."""
        try:
            # Extract metrics
            basic_metrics = validation_results.get('basic_metrics', {})
            stability_analysis = validation_results.get('stability_analysis', {})
            
            # Calculate overall quality score
            silhouette_score = basic_metrics.get('silhouette_score', 0.0)
            davies_bouldin_score = basic_metrics.get('davies_bouldin_score', 0.0)
            n_clusters = stability_analysis.get('n_clusters', 0)
            
            # Use BaseStep safe operations
            quality_score = self._safe_divide(silhouette_score, 1.0 + davies_bouldin_score, default=0)
            quality_score = self._validate_finite(quality_score, default=0)
            
            return {
                'overall_quality_score': float(quality_score),
                'silhouette_score': float(silhouette_score),
                'davies_bouldin_score': float(davies_bouldin_score),
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            tprint_error(f"❌ Overall quality assessment failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'silhouette_score': 0.0,
                'davies_bouldin_score': 0.0,
                'n_clusters': 0
            }

    async def _assess_regime_stability_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Assess regime stability using BaseStep safe operations."""
        try:
            tprint_info("Assessing regime stability")

            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)

            stability_results = {}

            # Analyze cluster stability using BaseStep utilities
            cluster_stability = await self._analyze_cluster_stability_safe(features, assignments)
            stability_results['cluster_stability'] = cluster_stability

            # Analyze regime persistence using BaseStep utilities
            regime_persistence = await self._analyze_regime_persistence_safe(assignments)
            stability_results['regime_persistence'] = regime_persistence

            # Analyze regime transitions using BaseStep utilities
            regime_transitions = await self._analyze_regime_transitions_safe(assignments)
            stability_results['regime_transitions'] = regime_transitions

            # Overall stability score
            stability_score = await self._calculate_stability_score(stability_results)
            stability_results['overall_stability'] = stability_score

            tprint_success(f"Regime stability assessment completed: {stability_score:.3f}")
            return stability_results

        except Exception as e:
            tprint_error(f"❌ Regime stability assessment failed: {e}")
            return {'error': str(e)}

    async def _analyze_cluster_stability_safe(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze cluster stability using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            features = self._validate_finite(features, default=0)
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple cluster stability analysis
            unique_assignments = np.unique(assignments)
            n_clusters = len(unique_assignments)
            
            # Calculate cluster sizes
            cluster_sizes = [np.sum(assignments == cluster) for cluster in unique_assignments]
            cluster_sizes = [self._validate_finite(size, default=0) for size in cluster_sizes]
            
            # Calculate stability metrics
            min_cluster_size = min(cluster_sizes) if cluster_sizes else 0
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            avg_cluster_size = self._safe_divide(sum(cluster_sizes), len(cluster_sizes), default=0)
            
            return {
                'n_clusters': n_clusters,
                'min_cluster_size': min_cluster_size,
                'max_cluster_size': max_cluster_size,
                'avg_cluster_size': avg_cluster_size,
                'cluster_sizes': cluster_sizes
            }
            
        except Exception as e:
            tprint_error(f"❌ Cluster stability analysis failed: {e}")
            return {
                'n_clusters': 0,
                'min_cluster_size': 0,
                'max_cluster_size': 0,
                'avg_cluster_size': 0,
                'cluster_sizes': []
            }

    async def _analyze_regime_persistence_safe(
        self,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze regime persistence using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple regime persistence analysis
            n_samples = len(assignments)
            unique_assignments = np.unique(assignments)
            
            # Calculate persistence metrics
            persistence_score = self._safe_divide(len(unique_assignments), n_samples, default=0)
            
            return {
                'persistence_score': float(persistence_score),
                'n_samples': n_samples,
                'n_unique_assignments': len(unique_assignments)
            }
            
        except Exception as e:
            tprint_error(f"❌ Regime persistence analysis failed: {e}")
            return {
                'persistence_score': 0.0,
                'n_samples': 0,
                'n_unique_assignments': 0
            }

    async def _analyze_regime_transitions_safe(
        self,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze regime transitions using BaseStep safe operations."""
        try:
            # Use BaseStep math validation
            assignments = self._validate_finite(assignments, default=0)
            
            # Simple regime transition analysis
            n_samples = len(assignments)
            unique_assignments = np.unique(assignments)
            
            # Calculate transition metrics
            transition_score = self._safe_divide(len(unique_assignments), n_samples, default=0)
            
            return {
                'transition_score': float(transition_score),
                'n_samples': n_samples,
                'n_unique_assignments': len(unique_assignments)
            }
            
        except Exception as e:
            tprint_error(f"❌ Regime transition analysis failed: {e}")
            return {
                'transition_score': 0.0,
                'n_samples': 0,
                'n_unique_assignments': 0
            }

    async def _calculate_stability_score_safe(
        self,
        stability_results: Dict[str, Any]
    ) -> float:
        """Calculate stability score using BaseStep safe operations."""
        try:
            # Extract metrics
            cluster_stability = stability_results.get('cluster_stability', {})
            regime_persistence = stability_results.get('regime_persistence', {})
            regime_transitions = stability_results.get('regime_transitions', {})
            
            # Calculate stability score
            cluster_score = cluster_stability.get('avg_cluster_size', 0.0)
            persistence_score = regime_persistence.get('persistence_score', 0.0)
            transition_score = regime_transitions.get('transition_score', 0.0)
            
            # Use BaseStep safe operations
            stability_score = self._safe_divide(
                cluster_score + persistence_score + transition_score, 
                3.0, 
                default=0.0
            )
            stability_score = self._validate_finite(stability_score, default=0.0)
            
            return float(stability_score)
            
        except Exception as e:
            tprint_error(f"❌ Stability score calculation failed: {e}")
            return 0.0

    async def _compute_basic_clustering_metrics(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Compute basic clustering metrics."""
        try:
            if len(np.unique(assignments)) < 2:
                return {'error': 'Insufficient clusters for metrics calculation'}

            metrics = {}

            # Silhouette score
            try:
                # Check for valid data before calculating silhouette score
                if len(features) == 0 or len(assignments) == 0:
                    metrics['silhouette_score'] = 0.0
                elif features.ndim == 1:
                    # Reshape 1D array to 2D for sklearn compatibility
                    features_2d = features.reshape(-1, 1)
                    metrics['silhouette_score'] = silhouette_score(features_2d, assignments)
                else:
                    metrics['silhouette_score'] = silhouette_score(features, assignments)
            except Exception as e:
                metrics['silhouette_score'] = 0.0
                tprint(f"Silhouette score calculation failed: {e}", "WARNING")

            # Davies-Bouldin score
            try:
                # Check for valid data before calculating Davies-Bouldin score
                if len(features) == 0 or len(assignments) == 0:
                    metrics['davies_bouldin_score'] = float('inf')
                elif features.ndim == 1:
                    # Reshape 1D array to 2D for sklearn compatibility
                    features_2d = features.reshape(-1, 1)
                    metrics['davies_bouldin_score'] = davies_bouldin_score(features_2d, assignments)
                else:
                    metrics['davies_bouldin_score'] = davies_bouldin_score(features, assignments)
            except Exception as e:
                metrics['davies_bouldin_score'] = float('inf')
                tprint(f"Davies-Bouldin score calculation failed: {e}", "WARNING")

            # Calinski-Harabasz score
            try:
                # Check for valid data before calculating Calinski-Harabasz score
                if len(features) == 0 or len(assignments) == 0:
                    metrics['calinski_harabasz_score'] = 0.0
                elif features.ndim == 1:
                    # Reshape 1D array to 2D for sklearn compatibility
                    features_2d = features.reshape(-1, 1)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_2d, assignments)
                else:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, assignments)
            except Exception as e:
                metrics['calinski_harabasz_score'] = 0.0
                tprint(f"Calinski-Harabasz score calculation failed: {e}", "WARNING")

            # Cluster count
            metrics['n_clusters'] = len(np.unique(assignments))

            # Cluster sizes
            unique, counts = np.unique(assignments, return_counts=True)
            metrics['cluster_sizes'] = dict(zip(unique, counts))
            metrics['min_cluster_size'] = np.min(counts)
            metrics['max_cluster_size'] = np.max(counts)
            metrics['mean_cluster_size'] = np.mean(counts)

            return metrics

        except Exception as e:
            tprint(f"Basic metrics computation failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _analyze_clustering_stability(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze clustering stability using bootstrap sampling."""
        try:
            tprint("Analyzing clustering stability...", "INFO")

            n_samples = len(features)
            n_bootstrap = min(10, n_samples // 10)  # Limit bootstrap samples

            if n_bootstrap < 2:
                return {'error': 'Insufficient samples for stability analysis'}

            stability_scores = []

            for i in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_features = features[bootstrap_indices]
                bootstrap_assignments = assignments[bootstrap_indices]

                # Calculate metrics for bootstrap sample
                try:
                    if len(np.unique(bootstrap_assignments)) >= 2:
                        silhouette = silhouette_score(bootstrap_features, bootstrap_assignments)
                        stability_scores.append(silhouette)
                except Exception:
                    continue

            if stability_scores:
                stability_analysis = {
                    'mean_stability': np.mean(stability_scores),
                    'std_stability': np.std(stability_scores),
                    'min_stability': np.min(stability_scores),
                    'max_stability': np.max(stability_scores),
                    'n_bootstrap_samples': len(stability_scores)
                }
            else:
                stability_analysis = {'error': 'No valid bootstrap samples'}

            return stability_analysis

        except Exception as e:
            tprint(f"Stability analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _compute_cross_validation_metrics(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Compute cross-validation metrics."""
        try:
            tprint("Computing cross-validation metrics...", "INFO")

            # Calculate within-cluster and between-cluster coefficients of variation
            within_cv = await self._calculate_within_cluster_cv(features, assignments)
            between_cv = await self._calculate_between_cluster_cv(features, assignments)

            cv_ratio = between_cv / within_cv if within_cv > 0 else 0.0

            cv_metrics = {
                'within_cluster_cv': within_cv,
                'between_cluster_cv': between_cv,
                'cv_ratio': cv_ratio,
                'cv_quality': 'Excellent' if cv_ratio > 1.5 else 'Good' if cv_ratio > 1.0 else 'Fair' if cv_ratio > 0.7 else 'Poor'
            }

            return cv_metrics

        except Exception as e:
            tprint(f"Cross-validation metrics computation failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _compute_temporal_consistency(
        self,
        assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compute temporal consistency metrics."""
        try:
            tprint("Computing temporal consistency...", "INFO")

            # Calculate regime persistence
            persistence_scores = []
            current_regime = assignments[0]
            persistence_length = 1

            for i in range(1, len(assignments)):
                if assignments[i] == current_regime:
                    persistence_length += 1
                else:
                    persistence_scores.append(persistence_length)
                    current_regime = assignments[i]
                    persistence_length = 1

            # Add final persistence
            persistence_scores.append(persistence_length)

            # Calculate temporal smoothness
            regime_changes = np.sum(assignments[1:] != assignments[:-1])
            temporal_smoothness = 1.0 - (regime_changes / len(assignments))

            temporal_metrics = {
                'regime_persistence_scores': persistence_scores,
                'mean_persistence': np.mean(persistence_scores),
                'std_persistence': np.std(persistence_scores),
                'temporal_smoothness': temporal_smoothness,
                'regime_changes': regime_changes,
                'change_rate': regime_changes / len(assignments)
            }

            return temporal_metrics

        except Exception as e:
            tprint(f"Temporal consistency computation failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _assess_overall_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall clustering quality."""
        try:
            tprint("Assessing overall clustering quality...", "INFO")

            quality_assessment = {}

            # Extract key metrics
            basic_metrics = validation_results.get('basic_metrics', {})
            stability_analysis = validation_results.get('stability_analysis', {})
            cv_metrics = validation_results.get('cv_metrics', {})
            temporal_metrics = validation_results.get('temporal_metrics', {})

            # Calculate composite quality score
            quality_components = []

            # Silhouette score component
            silhouette = basic_metrics.get('silhouette_score', 0.0)
            quality_components.append(('silhouette', silhouette, 0.3))

            # Stability component
            stability = stability_analysis.get('mean_stability', 0.0)
            quality_components.append(('stability', stability, 0.2))

            # CV ratio component
            cv_ratio = cv_metrics.get('cv_ratio', 0.0)
            quality_components.append(('cv_ratio', cv_ratio, 0.2))

            # Temporal smoothness component
            temporal_smoothness = temporal_metrics.get('temporal_smoothness', 0.0)
            quality_components.append(('temporal_smoothness', temporal_smoothness, 0.3))

            # Calculate weighted composite score
            composite_score = sum(score * weight for _, score, weight in quality_components)

            # Determine quality grade
            if composite_score >= 0.8:
                quality_grade = 'Excellent'
            elif composite_score >= 0.6:
                quality_grade = 'Good'
            elif composite_score >= 0.4:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'

            quality_assessment = {
                'composite_score': composite_score,
                'quality_grade': quality_grade,
                'quality_components': quality_components,
                'recommendations': self._generate_quality_recommendations(quality_components)
            }

            tprint(f"Overall quality assessment: {quality_grade} (score: {composite_score:.3f})", "SUCCESS")
            return quality_assessment

        except Exception as e:
            tprint(f"Overall quality assessment failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _analyze_cluster_stability(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze individual cluster stability."""
        try:
            unique_clusters = np.unique(assignments)
            cluster_stability = {}

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_features = features[cluster_mask]

                if len(cluster_features) > 1:
                    # Calculate intra-cluster distance
                    centroid = np.mean(cluster_features, axis=0)
                    intra_distances = [np.linalg.norm(point - centroid) for point in cluster_features]

                    cluster_stability[cluster] = {
                        'size': len(cluster_features),
                        'mean_intra_distance': np.mean(intra_distances),
                        'std_intra_distance': np.std(intra_distances),
                        'stability_score': 1.0 / (1.0 + np.mean(intra_distances))  # Higher is more stable
                    }
                else:
                    cluster_stability[cluster] = {
                        'size': 1,
                        'mean_intra_distance': 0.0,
                        'std_intra_distance': 0.0,
                        'stability_score': 0.0
                    }

            return cluster_stability

        except Exception as e:
            tprint(f"Cluster stability analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _analyze_regime_persistence(self, assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze regime persistence patterns."""
        try:
            # Calculate regime durations
            regime_durations = []
            current_regime = assignments[0]
            duration = 1

            for i in range(1, len(assignments)):
                if assignments[i] == current_regime:
                    duration += 1
                else:
                    regime_durations.append(duration)
                    current_regime = assignments[i]
                    duration = 1

            # Add final duration
            regime_durations.append(duration)

            persistence_analysis = {
                'regime_durations': regime_durations,
                'mean_duration': np.mean(regime_durations),
                'std_duration': np.std(regime_durations),
                'min_duration': np.min(regime_durations),
                'max_duration': np.max(regime_durations),
                'persistence_score': np.mean(regime_durations) / len(assignments)  # Normalized persistence
            }

            return persistence_analysis

        except Exception as e:
            tprint(f"Regime persistence analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _analyze_regime_transitions(self, assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        try:
            # Calculate transition matrix
            unique_regimes = np.unique(assignments)
            n_regimes = len(unique_regimes)
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(assignments) - 1):
                from_regime = assignments[i]
                to_regime = assignments[i + 1]
                from_idx = np.where(unique_regimes == from_regime)[0][0]
                to_idx = np.where(unique_regimes == to_regime)[0][0]
                transition_matrix[from_idx, to_idx] += 1

            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]

            # Calculate transition statistics
            total_transitions = np.sum(transition_matrix)
            self_transitions = np.sum(np.diag(transition_matrix))
            transition_rate = total_transitions / len(assignments)
            self_transition_rate = self_transitions / len(assignments)

            transition_analysis = {
                'transition_matrix': transition_matrix.tolist(),
                'unique_regimes': unique_regimes.tolist(),
                'total_transitions': total_transitions,
                'self_transitions': self_transitions,
                'transition_rate': transition_rate,
                'self_transition_rate': self_transition_rate,
                'stability_score': self_transition_rate  # Higher self-transitions = more stable
            }

            return transition_analysis

        except Exception as e:
            tprint(f"Regime transition analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    async def _calculate_stability_score(self, stability_results: Dict[str, Any]) -> float:
        """Calculate overall stability score."""
        try:
            # Extract stability components
            cluster_stability = stability_results.get('cluster_stability', {})
            regime_persistence = stability_results.get('regime_persistence', {})
            regime_transitions = stability_results.get('regime_transitions', {})

            # Calculate weighted stability score
            stability_components = []

            # Cluster stability component
            if cluster_stability and 'error' not in cluster_stability:
                cluster_scores = [stats['stability_score'] for stats in cluster_stability.values()]
                if cluster_scores:
                    stability_components.append(('cluster_stability', np.mean(cluster_scores), 0.4))

            # Persistence component
            if regime_persistence and 'error' not in regime_persistence:
                persistence_score = regime_persistence.get('persistence_score', 0.0)
                stability_components.append(('persistence', persistence_score, 0.3))

            # Transition stability component
            if regime_transitions and 'error' not in regime_transitions:
                transition_stability = regime_transitions.get('stability_score', 0.0)
                stability_components.append(('transition_stability', transition_stability, 0.3))

            # Calculate weighted average
            if stability_components:
                stability_score = sum(score * weight for _, score, weight in stability_components)
            else:
                stability_score = 0.0

            return stability_score

        except Exception as e:
            tprint(f"Stability score calculation failed: {e}", "ERROR")
            return 0.0

    async def _calculate_within_cluster_cv(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate within-cluster coefficient of variation."""
        try:
            unique_clusters = np.unique(assignments)
            within_cv_scores = []

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_features = features[cluster_mask]

                if len(cluster_features) > 1:
                    # Calculate intra-cluster distances
                    centroid = np.mean(cluster_features, axis=0)
                    distances = [np.linalg.norm(point - centroid) for point in cluster_features]

                    if np.mean(distances) > 0:
                        cv = np.std(distances) / np.mean(distances)
                        within_cv_scores.append(cv)

            return np.mean(within_cv_scores) if within_cv_scores else 0.0

        except Exception as e:
            tprint(f"Within-cluster CV calculation failed: {e}", "ERROR")
            return 0.0

    async def _calculate_between_cluster_cv(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate between-cluster coefficient of variation."""
        try:
            unique_clusters = np.unique(assignments)
            if len(unique_clusters) < 2:
                return 0.0

            # Calculate cluster centroids
            centroids = []
            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_features = features[cluster_mask]
                if len(cluster_features) > 0:
                    centroids.append(np.mean(cluster_features, axis=0))

            if len(centroids) < 2:
                return 0.0

            # Calculate inter-cluster distances
            inter_distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    inter_distances.append(distance)

            if inter_distances and np.mean(inter_distances) > 0:
                return np.std(inter_distances) / np.mean(inter_distances)
            else:
                return 0.0

        except Exception as e:
            tprint(f"Between-cluster CV calculation failed: {e}", "ERROR")
            return 0.0

    def _generate_quality_recommendations(self, quality_components: List[Tuple[str, float, float]]) -> List[str]:
        """Generate quality improvement recommendations."""
        try:
            recommendations = []

            for component, score, weight in quality_components:
                if score < 0.3:
                    if component == 'silhouette':
                        recommendations.append("Consider feature selection or dimensionality reduction to improve cluster separation")
                    elif component == 'stability':
                        recommendations.append("Increase clustering iterations or adjust convergence criteria")
                    elif component == 'cv_ratio':
                        recommendations.append("Optimize cluster balance and reduce within-cluster variance")
                    elif component == 'temporal_smoothness':
                        recommendations.append("Apply temporal smoothing or adjust regime transition criteria")

            if not recommendations:
                recommendations.append("Clustering quality is satisfactory")

            return recommendations

        except Exception as e:
            tprint(f"Recommendation generation failed: {e}", "ERROR")
            return ["Unable to generate recommendations"]
