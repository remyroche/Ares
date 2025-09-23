"""
NAS clustering metrics and evaluation.

This module provides comprehensive metrics for evaluating
NAS-driven clustering performance and regime quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time

logger = logging.getLogger(__name__)


@dataclass
class NASMetricsResult:
    """Result of NAS metrics calculation."""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    nas_score: float
    economic_significance_score: float
    trading_viability_score: float
    regime_stability_score: float
    regime_separation_score: float
    regime_consistency_score: float
    micro_regime_detection_accuracy: float
    execution_time: float
    metadata: Dict[str, Any]


class NASMetrics:
    """Metrics calculator for NAS-driven clustering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics thresholds
        self.silhouette_threshold = config.get('silhouette_threshold', 0.3)
        self.nas_score_threshold = config.get('nas_score_threshold', 0.4)
        self.economic_significance_threshold = config.get('economic_significance_threshold', 0.7)
        self.trading_viability_threshold = config.get('trading_viability_threshold', 0.6)
        
        self.logger.info("✅ NAS Metrics calculator initialized")
    
    def calculate_metrics(self, features: np.ndarray, labels: np.ndarray,
                         economic_scores: Optional[np.ndarray] = None,
                         trading_scores: Optional[np.ndarray] = None,
                         micro_regime_accuracy: Optional[float] = None) -> NASMetricsResult:
        """Calculate comprehensive NAS clustering metrics.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            economic_scores: Optional economic significance scores
            trading_scores: Optional trading viability scores
            micro_regime_accuracy: Optional micro-regime detection accuracy
            
        Returns:
            NASMetricsResult with calculated metrics
        """
        start_time = time.time()
        
        try:
            # Standard clustering metrics
            silhouette = self._calculate_silhouette_score(features, labels)
            calinski_harabasz = self._calculate_calinski_harabasz_score(features, labels)
            davies_bouldin = self._calculate_davies_bouldin_score(features, labels)
            
            # NAS-specific metrics
            nas_score = self._calculate_nas_score(features, labels)
            regime_stability = self._calculate_regime_stability(labels)
            regime_separation = self._calculate_regime_separation(features, labels)
            regime_consistency = self._calculate_regime_consistency(features, labels)
            
            # Economic and trading metrics
            economic_significance = self._calculate_economic_significance_score(
                economic_scores, labels
            )
            trading_viability = self._calculate_trading_viability_score(
                trading_scores, labels
            )
            
            # Micro-regime metrics
            micro_regime_accuracy = micro_regime_accuracy or 0.0
            
            execution_time = time.time() - start_time
            
            # Create result
            result = NASMetricsResult(
                silhouette_score=silhouette,
                calinski_harabasz_score=calinski_harabasz,
                davies_bouldin_score=davies_bouldin,
                nas_score=nas_score,
                economic_significance_score=economic_significance,
                trading_viability_score=trading_viability,
                regime_stability_score=regime_stability,
                regime_separation_score=regime_separation,
                regime_consistency_score=regime_consistency,
                micro_regime_detection_accuracy=micro_regime_accuracy,
                execution_time=execution_time,
                metadata={
                    'n_samples': len(features),
                    'n_features': features.shape[1],
                    'n_clusters': len(np.unique(labels)),
                    'thresholds': {
                        'silhouette': self.silhouette_threshold,
                        'nas_score': self.nas_score_threshold,
                        'economic_significance': self.economic_significance_threshold,
                        'trading_viability': self.trading_viability_threshold
                    }
                }
            )
            
            self.logger.info(f"✅ NAS metrics calculated in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS metrics calculation failed: {e}")
            return NASMetricsResult(
                silhouette_score=0.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=0.0,
                nas_score=0.0,
                economic_significance_score=0.0,
                trading_viability_score=0.0,
                regime_stability_score=0.0,
                regime_separation_score=0.0,
                regime_consistency_score=0.0,
                micro_regime_detection_accuracy=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            return silhouette_score(features, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Silhouette score calculation failed: {e}")
            return 0.0
    
    def _calculate_calinski_harabasz_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            return calinski_harabasz_score(features, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Calinski-Harabasz score calculation failed: {e}")
            return 0.0
    
    def _calculate_davies_bouldin_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score."""
        try:
            if len(np.unique(labels)) < 2:
                return 0.0
            return davies_bouldin_score(features, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Davies-Bouldin score calculation failed: {e}")
            return 0.0
    
    def _calculate_nas_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate custom NAS score."""
        try:
            # Combine multiple metrics for NAS score
            silhouette = self._calculate_silhouette_score(features, labels)
            calinski_harabasz = self._calculate_calinski_harabasz_score(features, labels)
            davies_bouldin = self._calculate_davies_bouldin_score(features, labels)
            
            # Normalize scores
            silhouette_norm = max(0, silhouette)
            calinski_harabasz_norm = max(0, min(1, calinski_harabasz / 1000))  # Normalize to 0-1
            davies_bouldin_norm = max(0, 1 - davies_bouldin)  # Invert (lower is better)
            
            # Weighted combination
            nas_score = (0.4 * silhouette_norm + 0.3 * calinski_harabasz_norm + 0.3 * davies_bouldin_norm)
            
            return nas_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS score calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            if len(labels) < 2:
                return 0.0
            
            # Calculate regime changes
            regime_changes = np.sum(np.diff(labels) != 0)
            total_periods = len(labels) - 1
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return stability
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime separation score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate inter-cluster distances
            inter_cluster_distances = []
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i < j:
                        cluster1_mask = labels == label1
                        cluster2_mask = labels == label2
                        
                        if np.any(cluster1_mask) and np.any(cluster2_mask):
                            center1 = np.mean(features[cluster1_mask], axis=0)
                            center2 = np.mean(features[cluster2_mask], axis=0)
                            distance = np.linalg.norm(center1 - center2)
                            inter_cluster_distances.append(distance)
            
            # Calculate intra-cluster distances
            intra_cluster_distances = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    center = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    intra_cluster_distances.extend(distances)
            
            # Calculate separation ratio
            if inter_cluster_distances and intra_cluster_distances:
                avg_inter = np.mean(inter_cluster_distances)
                avg_intra = np.mean(intra_cluster_distances)
                separation = avg_inter / (avg_intra + 1e-8)
                return min(separation, 1.0)  # Cap at 1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime separation calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_consistency(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate regime consistency score."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate consistency within each regime
            consistency_scores = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.any(cluster_mask):
                    cluster_features = features[cluster_mask]
                    # Calculate feature variance within cluster
                    feature_variance = np.var(cluster_features, axis=0)
                    # Lower variance = higher consistency
                    consistency = 1.0 / (1.0 + np.mean(feature_variance))
                    consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime consistency calculation failed: {e}")
            return 0.0
    
    def _calculate_economic_significance_score(self, economic_scores: Optional[np.ndarray],
                                             labels: np.ndarray) -> float:
        """Calculate economic significance score."""
        try:
            if economic_scores is None or len(economic_scores) == 0:
                return 0.0
            
            # Calculate mean economic significance
            mean_economic_significance = np.mean(economic_scores)
            
            # Calculate regime-wise economic significance
            unique_labels = np.unique(labels)
            regime_economic_scores = []
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_economic_score = np.mean(economic_scores[regime_mask])
                    regime_economic_scores.append(regime_economic_score)
            
            # Combine overall and regime-wise scores
            if regime_economic_scores:
                regime_economic_significance = np.mean(regime_economic_scores)
                combined_score = (mean_economic_significance + regime_economic_significance) / 2.0
            else:
                combined_score = mean_economic_significance
            
            return combined_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic significance score calculation failed: {e}")
            return 0.0
    
    def _calculate_trading_viability_score(self, trading_scores: Optional[np.ndarray],
                                         labels: np.ndarray) -> float:
        """Calculate trading viability score."""
        try:
            if trading_scores is None or len(trading_scores) == 0:
                return 0.0
            
            # Calculate mean trading viability
            mean_trading_viability = np.mean(trading_scores)
            
            # Calculate regime-wise trading viability
            unique_labels = np.unique(labels)
            regime_trading_scores = []
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_trading_score = np.mean(trading_scores[regime_mask])
                    regime_trading_scores.append(regime_trading_score)
            
            # Combine overall and regime-wise scores
            if regime_trading_scores:
                regime_trading_viability = np.mean(regime_trading_scores)
                combined_score = (mean_trading_viability + regime_trading_viability) / 2.0
            else:
                combined_score = mean_trading_viability
            
            return combined_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading viability score calculation failed: {e}")
            return 0.0
    
    def evaluate_clustering_quality(self, metrics_result: NASMetricsResult) -> Dict[str, Any]:
        """Evaluate clustering quality based on metrics.
        
        Args:
            metrics_result: NAS metrics result
            
        Returns:
            Dictionary with quality evaluation
        """
        try:
            # Evaluate individual metrics
            silhouette_quality = 'High' if metrics_result.silhouette_score > 0.5 else 'Medium' if metrics_result.silhouette_score > 0.3 else 'Low'
            nas_quality = 'High' if metrics_result.nas_score > 0.7 else 'Medium' if metrics_result.nas_score > 0.4 else 'Low'
            economic_quality = 'High' if metrics_result.economic_significance_score > 0.7 else 'Medium' if metrics_result.economic_significance_score > 0.5 else 'Low'
            trading_quality = 'High' if metrics_result.trading_viability_score > 0.7 else 'Medium' if metrics_result.trading_viability_score > 0.5 else 'Low'
            stability_quality = 'High' if metrics_result.regime_stability_score > 0.8 else 'Medium' if metrics_result.regime_stability_score > 0.6 else 'Low'
            separation_quality = 'High' if metrics_result.regime_separation_score > 0.7 else 'Medium' if metrics_result.regime_separation_score > 0.5 else 'Low'
            consistency_quality = 'High' if metrics_result.regime_consistency_score > 0.7 else 'Medium' if metrics_result.regime_consistency_score > 0.5 else 'Low'
            micro_regime_quality = 'High' if metrics_result.micro_regime_detection_accuracy > 0.8 else 'Medium' if metrics_result.micro_regime_detection_accuracy > 0.6 else 'Low'
            
            # Overall quality assessment
            quality_scores = [
                metrics_result.silhouette_score,
                metrics_result.nas_score,
                metrics_result.economic_significance_score,
                metrics_result.trading_viability_score,
                metrics_result.regime_stability_score,
                metrics_result.regime_separation_score,
                metrics_result.regime_consistency_score,
                metrics_result.micro_regime_detection_accuracy
            ]
            
            overall_quality = np.mean(quality_scores)
            overall_quality_level = 'High' if overall_quality > 0.7 else 'Medium' if overall_quality > 0.5 else 'Low'
            
            # Recommendations
            recommendations = []
            if metrics_result.silhouette_score < self.silhouette_threshold:
                recommendations.append("Consider adjusting clustering parameters to improve silhouette score")
            if metrics_result.nas_score < self.nas_score_threshold:
                recommendations.append("Consider optimizing NAS architecture for better regime detection")
            if metrics_result.economic_significance_score < self.economic_significance_threshold:
                recommendations.append("Consider enhancing economic significance features")
            if metrics_result.trading_viability_score < self.trading_viability_threshold:
                recommendations.append("Consider improving trading viability features")
            if metrics_result.regime_stability_score < 0.6:
                recommendations.append("Consider increasing regime stability constraints")
            if metrics_result.regime_separation_score < 0.5:
                recommendations.append("Consider improving regime separation")
            if metrics_result.regime_consistency_score < 0.5:
                recommendations.append("Consider enhancing regime consistency")
            if metrics_result.micro_regime_detection_accuracy < 0.6:
                recommendations.append("Consider improving micro-regime detection")
            
            return {
                'overall_quality': overall_quality_level,
                'overall_score': overall_quality,
                'individual_qualities': {
                    'silhouette': silhouette_quality,
                    'nas_score': nas_quality,
                    'economic_significance': economic_quality,
                    'trading_viability': trading_quality,
                    'regime_stability': stability_quality,
                    'regime_separation': separation_quality,
                    'regime_consistency': consistency_quality,
                    'micro_regime_detection': micro_regime_quality
                },
                'recommendations': recommendations,
                'quality_scores': quality_scores
            }
            
        except Exception as e:
            self.logger.error(f"❌ Clustering quality evaluation failed: {e}")
            return {
                'overall_quality': 'Unknown',
                'overall_score': 0.0,
                'individual_qualities': {},
                'recommendations': [f"Error in quality evaluation: {str(e)}"],
                'quality_scores': []
            }


class NASClusteringMetrics:
    """Enhanced metrics for NAS clustering evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS clustering metrics.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize base metrics
        self.nas_metrics = NASMetrics(config)
        
        self.logger.info("✅ NAS Clustering Metrics initialized")
    
    def calculate_comprehensive_metrics(self, features: np.ndarray, labels: np.ndarray,
                                      economic_scores: Optional[np.ndarray] = None,
                                      trading_scores: Optional[np.ndarray] = None,
                                      micro_regime_accuracy: Optional[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive NAS clustering metrics.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            economic_scores: Optional economic significance scores
            trading_scores: Optional trading viability scores
            micro_regime_accuracy: Optional micro-regime detection accuracy
            
        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            # Calculate base metrics
            base_metrics = self.nas_metrics.calculate_metrics(
                features, labels, economic_scores, trading_scores, micro_regime_accuracy
            )
            
            # Calculate additional metrics
            additional_metrics = self._calculate_additional_metrics(
                features, labels, economic_scores, trading_scores
            )
            
            # Evaluate clustering quality
            quality_evaluation = self.nas_metrics.evaluate_clustering_quality(base_metrics)
            
            # Create comprehensive result
            comprehensive_metrics = {
                'base_metrics': base_metrics,
                'additional_metrics': additional_metrics,
                'quality_evaluation': quality_evaluation,
                'summary': self._create_metrics_summary(base_metrics, additional_metrics, quality_evaluation)
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive metrics calculation failed: {e}")
            return {
                'base_metrics': None,
                'additional_metrics': {},
                'quality_evaluation': {},
                'summary': {'error': str(e)}
            }
    
    def _calculate_additional_metrics(self, features: np.ndarray, labels: np.ndarray,
                                     economic_scores: Optional[np.ndarray] = None,
                                     trading_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate additional metrics for comprehensive evaluation."""
        try:
            additional_metrics = {}
            
            # Regime transition metrics
            additional_metrics['regime_transitions'] = self._calculate_regime_transition_metrics(labels)
            
            # Regime duration metrics
            additional_metrics['regime_durations'] = self._calculate_regime_duration_metrics(labels)
            
            # Feature importance metrics
            additional_metrics['feature_importance'] = self._calculate_feature_importance_metrics(features, labels)
            
            # Economic and trading regime analysis
            if economic_scores is not None:
                additional_metrics['economic_regime_analysis'] = self._calculate_economic_regime_analysis(
                    labels, economic_scores
                )
            
            if trading_scores is not None:
                additional_metrics['trading_regime_analysis'] = self._calculate_trading_regime_analysis(
                    labels, trading_scores
                )
            
            return additional_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Additional metrics calculation failed: {e}")
            return {}
    
    def _calculate_regime_transition_metrics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime transition metrics."""
        try:
            if len(labels) < 2:
                return {}
            
            # Calculate transition counts
            unique_labels = np.unique(labels)
            n_regimes = len(unique_labels)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                
                if current_regime in unique_labels and next_regime in unique_labels:
                    current_idx = np.where(unique_labels == current_regime)[0][0]
                    next_idx = np.where(unique_labels == next_regime)[0][0]
                    transition_matrix[current_idx, next_idx] += 1
            
            # Calculate transition probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            # Calculate transition entropy
            transition_entropy = 0.0
            for i in range(n_regimes):
                for j in range(n_regimes):
                    if transition_probs[i, j] > 0:
                        transition_entropy -= transition_probs[i, j] * np.log2(transition_probs[i, j])
            
            return {
                'transition_matrix': transition_matrix.tolist(),
                'transition_probabilities': transition_probs.tolist(),
                'transition_entropy': transition_entropy,
                'n_transitions': int(np.sum(transition_matrix)),
                'transition_rate': float(np.sum(transition_matrix) / (len(labels) - 1))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition metrics calculation failed: {e}")
            return {}
    
    def _calculate_regime_duration_metrics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime duration metrics."""
        try:
            if len(labels) == 0:
                return {}
            
            # Calculate regime durations
            regime_changes = np.diff(labels) != 0
            regime_starts = np.concatenate([[True], regime_changes])
            regime_ends = np.concatenate([regime_changes, [True]])
            
            regime_durations = []
            current_duration = 0
            
            for i, (start, end) in enumerate(zip(regime_starts, regime_ends)):
                if start:
                    current_duration = 1
                else:
                    current_duration += 1
                
                if end:
                    regime_durations.append(current_duration)
            
            return {
                'regime_durations': regime_durations,
                'mean_duration': float(np.mean(regime_durations)),
                'median_duration': float(np.median(regime_durations)),
                'std_duration': float(np.std(regime_durations)),
                'min_duration': int(np.min(regime_durations)),
                'max_duration': int(np.max(regime_durations)),
                'total_regime_changes': int(np.sum(regime_changes))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime duration metrics calculation failed: {e}")
            return {}
    
    def _calculate_feature_importance_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate feature importance metrics."""
        try:
            if features.size == 0:
                return {}
            
            # Calculate feature variance within clusters
            unique_labels = np.unique(labels)
            feature_importance = []
            
            for feature_idx in range(features.shape[1]):
                feature_values = features[:, feature_idx]
                feature_variance = np.var(feature_values)
                
                # Calculate within-cluster variance
                within_cluster_variance = 0.0
                for label in unique_labels:
                    cluster_mask = labels == label
                    if np.any(cluster_mask):
                        cluster_feature_values = feature_values[cluster_mask]
                        cluster_variance = np.var(cluster_feature_values)
                        within_cluster_variance += cluster_variance * np.sum(cluster_mask)
                
                within_cluster_variance /= len(feature_values)
                
                # Feature importance is inverse of within-cluster variance
                importance = 1.0 / (1.0 + within_cluster_variance)
                feature_importance.append(importance)
            
            return {
                'feature_importance': feature_importance,
                'top_features': np.argsort(feature_importance)[-10:].tolist(),  # Top 10 features
                'feature_variance': np.var(features, axis=0).tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance metrics calculation failed: {e}")
            return {}
    
    def _calculate_economic_regime_analysis(self, labels: np.ndarray, 
                                          economic_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate economic regime analysis."""
        try:
            unique_labels = np.unique(labels)
            economic_analysis = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_economic_scores = economic_scores[regime_mask]
                    economic_analysis[f'regime_{label}'] = {
                        'mean_economic_significance': float(np.mean(regime_economic_scores)),
                        'std_economic_significance': float(np.std(regime_economic_scores)),
                        'min_economic_significance': float(np.min(regime_economic_scores)),
                        'max_economic_significance': float(np.max(regime_economic_scores)),
                        'regime_size': int(np.sum(regime_mask))
                    }
            
            return economic_analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Economic regime analysis calculation failed: {e}")
            return {}
    
    def _calculate_trading_regime_analysis(self, labels: np.ndarray, 
                                         trading_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate trading regime analysis."""
        try:
            unique_labels = np.unique(labels)
            trading_analysis = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_trading_scores = trading_scores[regime_mask]
                    trading_analysis[f'regime_{label}'] = {
                        'mean_trading_viability': float(np.mean(regime_trading_scores)),
                        'std_trading_viability': float(np.std(regime_trading_scores)),
                        'min_trading_viability': float(np.min(regime_trading_scores)),
                        'max_trading_viability': float(np.max(regime_trading_scores)),
                        'regime_size': int(np.sum(regime_mask))
                    }
            
            return trading_analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trading regime analysis calculation failed: {e}")
            return {}
    
    def _create_metrics_summary(self, base_metrics: NASMetricsResult,
                              additional_metrics: Dict[str, Any],
                              quality_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Create metrics summary."""
        try:
            return {
                'overall_quality': quality_evaluation.get('overall_quality', 'Unknown'),
                'overall_score': quality_evaluation.get('overall_score', 0.0),
                'key_metrics': {
                    'silhouette_score': base_metrics.silhouette_score,
                    'nas_score': base_metrics.nas_score,
                    'economic_significance': base_metrics.economic_significance_score,
                    'trading_viability': base_metrics.trading_viability_score,
                    'regime_stability': base_metrics.regime_stability_score,
                    'regime_separation': base_metrics.regime_separation_score,
                    'regime_consistency': base_metrics.regime_consistency_score,
                    'micro_regime_accuracy': base_metrics.micro_regime_detection_accuracy
                },
                'recommendations': quality_evaluation.get('recommendations', []),
                'execution_time': base_metrics.execution_time,
                'metadata': base_metrics.metadata
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Metrics summary creation failed: {e}")
            return {'error': str(e)}