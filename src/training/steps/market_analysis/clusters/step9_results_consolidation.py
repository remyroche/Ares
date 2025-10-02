"""
Step 9: Results Consolidation for NAS-TAS Clustering.

This module handles the final results consolidation, artifact creation,
and comprehensive metrics calculation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import (
    calculate_consensus_metrics,
    calculate_disagreement_metrics,
    calculate_economic_scores,
    calculate_trading_scores,
    calculate_stability_scores,
    MetricsCalculator,
    get_logger
)
from .step1_feature_preparation import ClusteringContext


class ResultsConsolidationStep:
    """Step 9: Results consolidation and artifact creation."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the results consolidation step."""
        self.verbose = verbose
        self.logger = get_logger('ResultsConsolidationStep')
        self.metrics_calculator = MetricsCalculator(verbose=True)
        
    async def execute(self, context: ClusteringContext, config: Any) -> Dict[str, Any]:
        """Execute results consolidation step."""
        try:
            tprint("Step 9: Starting results consolidation...", "INFO")
            
            # Calculate comprehensive clustering metrics
            clustering_metrics = await self._calculate_clustering_metrics(context)
            
            # Generate cluster characteristics
            cluster_characteristics = await self._generate_cluster_characteristics(context)
            
            # Create consolidated artifacts
            artifacts = await self._create_consolidated_artifacts(context, config)
            
            # Summarize results
            final_results = await self._summarize_results(
                context, clustering_metrics, cluster_characteristics, artifacts
            )
            
            tprint("Step 9: Results consolidation completed successfully", "SUCCESS")
            return final_results
            
        except Exception as e:
            tprint(f"Step 9: Results consolidation failed: {e}", "ERROR")
            raise ValueError(f"Results consolidation failed: {e}")
    
    async def _calculate_clustering_metrics(self, context: ClusteringContext) -> Dict[str, Any]:
        """Calculate comprehensive clustering metrics using shared utilities."""
        try:
            tprint("Calculating comprehensive clustering metrics...", "INFO")
            
            features = context.optimized_features
            assignments = context.optimized_assignments
            market_data = context.market_data
            
            # Use shared metrics calculator
            metrics_result = self.metrics_calculator.calculate_all_metrics(
                features=features,
                assignments=assignments,
                market_data=market_data
            )
            
            # Add additional clustering-specific metrics
            additional_metrics = await self._calculate_additional_metrics(features, assignments)
            metrics_result.update(additional_metrics)
            
            tprint("Clustering metrics calculation completed", "SUCCESS")
            return metrics_result
            
        except Exception as e:
            tprint(f"Clustering metrics calculation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _generate_cluster_characteristics(self, context: ClusteringContext) -> Dict[str, Any]:
        """Generate cluster characteristics using shared utilities."""
        try:
            tprint("Generating cluster characteristics...", "INFO")
            
            features = context.optimized_features
            assignments = context.optimized_assignments
            market_data = context.market_data
            
            # Use shared characteristics generator
            from ..shared_utils import CharacteristicsGenerator
            characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            characteristics = characteristics_generator.generate_cluster_characteristics(
                features=features,
                assignments=assignments,
                market_data=market_data
            )
            
            tprint("Cluster characteristics generation completed", "SUCCESS")
            return characteristics
            
        except Exception as e:
            tprint(f"Cluster characteristics generation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _create_consolidated_artifacts(
        self, 
        context: ClusteringContext, 
        config: Any
    ) -> Dict[str, Any]:
        """Create consolidated artifacts for the clustering results."""
        try:
            tprint("Creating consolidated artifacts...", "INFO")
            
            artifacts = {}
            
            # Create regime assignments dataframe
            regime_df = await self._create_regime_assignments_dataframe(
                context.optimized_assignments,
                context.market_data,
                context.tas_assignments,
                context.nas_assignments
            )
            artifacts['regime_assignments'] = regime_df
            
            # Create clustering summary
            clustering_summary = await self._create_clustering_summary(context)
            artifacts['clustering_summary'] = clustering_summary
            
            # Create feature importance analysis
            feature_importance = await self._create_feature_importance_analysis(context)
            artifacts['feature_importance'] = feature_importance
            
            # Create regime transition analysis
            transition_analysis = await self._create_regime_transition_analysis(context)
            artifacts['transition_analysis'] = transition_analysis
            
            # Create performance metrics
            performance_metrics = await self._create_performance_metrics(context)
            artifacts['performance_metrics'] = performance_metrics
            
            tprint("Consolidated artifacts creation completed", "SUCCESS")
            return artifacts
            
        except Exception as e:
            tprint(f"Artifacts creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _summarize_results(
        self, 
        context: ClusteringContext, 
        clustering_metrics: Dict[str, Any],
        cluster_characteristics: Dict[str, Any],
        artifacts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize all results into a comprehensive output."""
        try:
            tprint("Summarizing final results...", "INFO")
            
            # Create comprehensive results dictionary
            final_results = {
                'clustering_result': {
                    'n_clusters': len(np.unique(context.optimized_assignments)),
                    'assignments': context.optimized_assignments.tolist(),
                    'feature_names': context.optimized_feature_names,
                    'optimal_k': context.optimal_k,
                    'final_k': context.final_k
                },
                'metrics': clustering_metrics,
                'characteristics': cluster_characteristics,
                'artifacts': artifacts,
                'validation_results': getattr(context, 'validation_results', {}),
                'stability_results': getattr(context, 'stability_results', {}),
                'execution_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'feature_count': context.optimized_features.shape[1],
                    'sample_count': context.optimized_features.shape[0],
                    'clustering_algorithm': 'NAS-TAS Clustering (Refactored)',
                    'version': '2.0.0'
                }
            }
            
            # Add quality assessment
            if 'quality_assessment' in clustering_metrics:
                final_results['quality_assessment'] = clustering_metrics['quality_assessment']
            
            # Add recommendations
            final_results['recommendations'] = await self._generate_final_recommendations(
                clustering_metrics, cluster_characteristics
            )
            
            tprint("Final results summarization completed", "SUCCESS")
            return final_results
            
        except Exception as e:
            tprint(f"Results summarization failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _calculate_additional_metrics(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate additional clustering-specific metrics."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            additional_metrics = {}
            
            # Basic clustering metrics
            if len(np.unique(assignments)) >= 2:
                additional_metrics['silhouette_score'] = silhouette_score(features, assignments)
                additional_metrics['davies_bouldin_score'] = davies_bouldin_score(features, assignments)
                additional_metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, assignments)
            
            # Cluster balance
            unique, counts = np.unique(assignments, return_counts=True)
            additional_metrics['cluster_balance'] = 1.0 - (np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0
            
            # Cluster sizes
            additional_metrics['cluster_sizes'] = dict(zip(unique, counts))
            additional_metrics['min_cluster_size'] = int(np.min(counts))
            additional_metrics['max_cluster_size'] = int(np.max(counts))
            additional_metrics['mean_cluster_size'] = float(np.mean(counts))
            
            return additional_metrics
            
        except Exception as e:
            tprint(f"Additional metrics calculation failed: {e}", "ERROR")
            return {}
    
    async def _create_regime_assignments_dataframe(
        self, 
        assignments: np.ndarray, 
        market_data: pd.DataFrame,
        tas_assignments: Optional[np.ndarray] = None,
        nas_assignments: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Create regime assignments dataframe."""
        try:
            # Create base dataframe
            regime_df = pd.DataFrame({
                'timestamp': market_data.index if hasattr(market_data, 'index') else range(len(assignments)),
                'regime_assignment': assignments,
                'tas_assignment': tas_assignments if tas_assignments is not None else assignments,
                'nas_assignment': nas_assignments if nas_assignments is not None else assignments
            })
            
            # Add regime labels
            regime_df['regime_label'] = regime_df['regime_assignment'].apply(
                lambda x: f"Regime_{x}"
            )
            
            # Add regime characteristics
            regime_df['regime_persistence'] = self._calculate_regime_persistence(assignments)
            regime_df['regime_confidence'] = self._calculate_regime_confidence(assignments)
            
            return regime_df
            
        except Exception as e:
            tprint(f"Regime assignments dataframe creation failed: {e}", "ERROR")
            return pd.DataFrame()
    
    async def _create_clustering_summary(self, context: ClusteringContext) -> Dict[str, Any]:
        """Create clustering summary."""
        try:
            assignments = context.optimized_assignments
            unique_regimes = np.unique(assignments)
            
            summary = {
                'total_samples': len(assignments),
                'n_regimes': len(unique_regimes),
                'regime_distribution': {
                    int(regime): int(np.sum(assignments == regime)) 
                    for regime in unique_regimes
                },
                'feature_count': context.optimized_features.shape[1],
                'feature_names': context.optimized_feature_names,
                'optimization_completed': True,
                'validation_completed': hasattr(context, 'validation_results')
            }
            
            return summary
            
        except Exception as e:
            tprint(f"Clustering summary creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _create_feature_importance_analysis(self, context: ClusteringContext) -> Dict[str, Any]:
        """Create feature importance analysis."""
        try:
            feature_importance = {}
            
            if hasattr(context, 'pca_loading_scores') and context.pca_loading_scores:
                feature_importance['pca_loadings'] = context.pca_loading_scores
            
            if hasattr(context, 'feature_scores') and context.feature_scores:
                feature_importance['feature_scores'] = context.feature_scores
            
            # Calculate feature-cluster correlations
            correlations = self._calculate_feature_cluster_correlations(
                context.optimized_features, 
                context.optimized_assignments
            )
            feature_importance['cluster_correlations'] = correlations
            
            return feature_importance
            
        except Exception as e:
            tprint(f"Feature importance analysis creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _create_regime_transition_analysis(self, context: ClusteringContext) -> Dict[str, Any]:
        """Create regime transition analysis."""
        try:
            assignments = context.optimized_assignments
            
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
            
            # Normalize
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
            transition_analysis = {
                'transition_matrix': transition_matrix.tolist(),
                'unique_regimes': unique_regimes.tolist(),
                'transition_counts': transition_matrix.sum(),
                'self_transition_rate': np.sum(np.diag(transition_matrix)) / len(assignments)
            }
            
            return transition_analysis
            
        except Exception as e:
            tprint(f"Regime transition analysis creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _create_performance_metrics(self, context: ClusteringContext) -> Dict[str, Any]:
        """Create performance metrics."""
        try:
            performance_metrics = {
                'execution_time': getattr(context, 'execution_time', 0.0),
                'memory_usage': getattr(context, 'memory_usage', 0.0),
                'convergence_iterations': getattr(context, 'convergence_iterations', 0),
                'optimization_trials': getattr(context, 'optimization_trials', 0)
            }
            
            return performance_metrics
            
        except Exception as e:
            tprint(f"Performance metrics creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    async def _generate_final_recommendations(
        self, 
        clustering_metrics: Dict[str, Any],
        cluster_characteristics: Dict[str, Any]
    ) -> List[str]:
        """Generate final recommendations based on results."""
        try:
            recommendations = []
            
            # Check clustering quality
            if 'quality_assessment' in clustering_metrics:
                quality_grade = clustering_metrics['quality_assessment'].get('quality_grade', 'Unknown')
                if quality_grade in ['Poor', 'Fair']:
                    recommendations.append("Consider adjusting clustering parameters or feature selection")
            
            # Check cluster balance
            cluster_balance = clustering_metrics.get('cluster_balance', 0.0)
            if cluster_balance < 0.5:
                recommendations.append("Cluster sizes are imbalanced - consider adjusting clustering algorithm")
            
            # Check regime stability
            if 'stability_results' in clustering_metrics:
                stability_score = clustering_metrics['stability_results'].get('overall_stability', 0.0)
                if stability_score < 0.5:
                    recommendations.append("Regime stability is low - consider temporal smoothing")
            
            if not recommendations:
                recommendations.append("Clustering results are satisfactory")
            
            return recommendations
            
        except Exception as e:
            tprint(f"Recommendation generation failed: {e}", "ERROR")
            return ["Unable to generate recommendations"]
    
    def _calculate_regime_persistence(self, assignments: np.ndarray) -> np.ndarray:
        """Calculate regime persistence for each sample."""
        try:
            persistence = np.zeros(len(assignments))
            current_regime = assignments[0]
            current_length = 1
        
            for i in range(1, len(assignments)):
                if assignments[i] == current_regime:
                    current_length += 1
                else:
                    # Update persistence for the previous regime
                    for j in range(i - current_length, i):
                        persistence[j] = current_length
                    current_regime = assignments[i]
                    current_length = 1
            
            # Update final regime
            for j in range(len(assignments) - current_length, len(assignments)):
                persistence[j] = current_length
            
            return persistence
            
        except Exception as e:
            tprint(f"Regime persistence calculation failed: {e}", "ERROR")
            return np.ones(len(assignments))
    
    def _calculate_regime_confidence(self, assignments: np.ndarray) -> np.ndarray:
        """Calculate regime confidence for each sample."""
        try:
            # Simple confidence based on local consistency
            confidence = np.ones(len(assignments))
            
            for i in range(len(assignments)):
                # Check local neighborhood consistency
                start_idx = max(0, i - 2)
                end_idx = min(len(assignments), i + 3)
                local_assignments = assignments[start_idx:end_idx]
                
                # Calculate consistency
                unique, counts = np.unique(local_assignments, return_counts=True)
                majority_count = np.max(counts)
                consistency = majority_count / len(local_assignments)
                
                confidence[i] = consistency
            
            return confidence
            
        except Exception as e:
            tprint(f"Regime confidence calculation failed: {e}", "ERROR")
            return np.ones(len(assignments))
    
    def _calculate_feature_cluster_correlations(
        self, 
        features: np.ndarray, 
        assignments: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature-cluster correlations."""
        try:
            correlations = {}
            unique_clusters = np.unique(assignments)
            
            for i, feature_idx in enumerate(range(features.shape[1])):
                feature_values = features[:, feature_idx]
                feature_correlations = []
                
                for cluster in unique_clusters:
                    cluster_mask = assignments == cluster
                    if np.sum(cluster_mask) > 1:
                        # Calculate correlation between feature and cluster membership
                        cluster_indicator = cluster_mask.astype(float)
                        correlation = np.corrcoef(feature_values, cluster_indicator)[0, 1]
                        if not np.isnan(correlation):
                            feature_correlations.append(abs(correlation))
                
                if feature_correlations:
                    correlations[f'feature_{i}'] = np.mean(feature_correlations)
                else:
                    correlations[f'feature_{i}'] = 0.0
            
            return correlations
            
        except Exception as e:
            tprint(f"Feature-cluster correlation calculation failed: {e}", "ERROR")
            return {}