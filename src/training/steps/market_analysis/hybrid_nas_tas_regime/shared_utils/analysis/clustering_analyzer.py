"""
Clustering Analyzer for Advanced Market Analysis.

This module provides clustering analysis capabilities that can be used
by both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from src.utils.logger import system_logger


@dataclass
class ClusteringAnalysisConfig:
    """Configuration for clustering analysis."""
    max_clusters: int = 10
    min_clusters: int = 2
    enable_cluster_validation: bool = True
    enable_cluster_characterization: bool = True
    enable_stability_analysis: bool = True
    validation_metrics: List[str] = field(default_factory=lambda: ['silhouette', 'calinski_harabasz', 'davies_bouldin'])


@dataclass
class ClusterCharacteristics:
    """Characteristics of a cluster."""
    cluster_id: int
    size: int
    centroid: np.ndarray
    variance: float
    silhouette_score: float
    stability_score: float
    dominant_features: List[str]
    market_regime: str
    risk_profile: Dict[str, float]


class ClusteringAnalyzer:
    """
    Clustering analyzer for market regime analysis.

    This class provides comprehensive clustering analysis including validation,
    characterization, and stability assessment that can be used by both NAS
    and TAS systems.
    """

    def __init__(self, config: ClusteringAnalysisConfig):
        """
        Initialize the clustering analyzer.

        Args:
            config: Clustering analysis configuration
        """
        self.logger = system_logger.getChild('ClusteringAnalyzer')
        self.config = config

        self.logger.info("✅ Clustering Analyzer initialized"
        self.logger.info(f"   Max clusters: {config.max_clusters}")
        self.logger.info(f"   Min clusters: {config.min_clusters}")

    def analyze_clustering(self,
                          data: pd.DataFrame,
                          cluster_labels: np.ndarray,
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive clustering analysis.

        Args:
            data: Data to analyze
            cluster_labels: Cluster labels for each data point
            feature_names: Names of features in data

        Returns:
            Dictionary with comprehensive clustering analysis
        """
        try:
            self.logger.info("📊 Performing comprehensive clustering analysis")

            # Basic cluster statistics
            cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)

            # Cluster validation
            validation_results = {}
            if self.config.enable_cluster_validation:
                validation_results = self._validate_clustering(data, cluster_labels)

            # Cluster characteristics
            cluster_characteristics = {}
            if self.config.enable_cluster_characterization:
                cluster_characteristics = self._analyze_cluster_characteristics(
                    data, cluster_labels, feature_names
                )

            # Stability analysis
            stability_analysis = {}
            if self.config.enable_stability_analysis:
                stability_analysis = self._analyze_cluster_stability(data, cluster_labels)

            analysis_result = {
                'cluster_statistics': cluster_stats,
                'validation_results': validation_results,
                'cluster_characteristics': cluster_characteristics,
                'stability_analysis': stability_analysis,
                'clustering_summary': self._generate_clustering_summary(
                    cluster_stats, validation_results, cluster_characteristics
                ),
                'analysis_metadata': {
                    'total_samples': len(data),
                    'total_features': len(data.columns),
                    'unique_clusters': len(set(cluster_labels)),
                    'analysis_timestamp': pd.Timestamp.now()
                }
            }

            self.logger.info(f"✅ Clustering analysis completed for {len(cluster_characteristics)} clusters")
            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Clustering analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_cluster_statistics(self,
                                    data: pd.DataFrame,
                                    cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate basic cluster statistics.

        Args:
            data: Data to analyze
            cluster_labels: Cluster labels

        Returns:
            Dictionary of cluster statistics
        """
        try:
            stats = {}
            unique_clusters = np.unique(cluster_labels)

            stats['total_clusters'] = len(unique_clusters)
            stats['total_samples'] = len(data)

            # Cluster sizes
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            stats['cluster_sizes'] = cluster_sizes.to_dict()
            stats['cluster_percentages'] = (cluster_sizes / len(data) * 100).to_dict()

            # Size statistics
            sizes = list(cluster_sizes.values)
            stats['min_cluster_size'] = int(np.min(sizes))
            stats['max_cluster_size'] = int(np.max(sizes))
            stats['mean_cluster_size'] = float(np.mean(sizes))
            stats['std_cluster_size'] = float(np.std(sizes))

            # Check for balanced clusters
            size_ratio = stats['min_cluster_size'] / stats['max_cluster_size']
            stats['balanced_clusters'] = size_ratio > 0.5  # More than 50% of max size

            # Check for sufficient samples per cluster
            min_samples_per_cluster = 10  # Minimum recommended
            stats['sufficient_samples'] = all(size >= min_samples_per_cluster for size in sizes)

            if not stats['sufficient_samples']:
                self.logger.warning(f"⚠️ Some clusters have insufficient samples (< {min_samples_per_cluster})")

            return stats

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster statistics calculation failed: {e}")
            return {}

    def _validate_clustering(self,
                           data: pd.DataFrame,
                           cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Validate clustering quality using multiple metrics.

        Args:
            data: Data to validate
            cluster_labels: Cluster labels

        Returns:
            Dictionary of validation results
        """
        try:
            validation = {}
            unique_clusters = len(set(cluster_labels))

            # Check minimum requirements
            if len(data) < 2 * unique_clusters or unique_clusters < 2:
                validation['error'] = 'Insufficient data or too few clusters for validation'
                return validation

            # Prepare data for validation
            X_scaled = StandardScaler().fit_transform(data)
            labels = np.array(cluster_labels)

            # Calculate validation metrics
            metrics = {}

            # Silhouette score
            if 'silhouette' in self.config.validation_metrics:
                try:
                    silhouette_avg = silhouette_score(X_scaled, labels)
                    metrics['silhouette_score'] = float(silhouette_avg)
                    metrics['silhouette_interpretation'] = self._interpret_silhouette_score(silhouette_avg)
                except Exception as e:
                    self.logger.warning(f"⚠️ Silhouette score calculation failed: {e}")
                    metrics['silhouette_score'] = None

            # Calinski-Harabasz score
            if 'calinski_harabasz' in self.config.validation_metrics:
                try:
                    ch_score = calinski_harabasz_score(X_scaled, labels)
                    metrics['calinski_harabasz_score'] = float(ch_score)
                    metrics['ch_interpretation'] = 'Higher is better' if ch_score > 0 else 'Invalid score'
                except Exception as e:
                    self.logger.warning(f"⚠️ Calinski-Harabasz score calculation failed: {e}")
                    metrics['calinski_harabasz_score'] = None

            # Davies-Bouldin score
            if 'davies_bouldin' in self.config.validation_metrics:
                try:
                    db_score = davies_bouldin_score(X_scaled, labels)
                    metrics['davies_bouldin_score'] = float(db_score)
                    metrics['db_interpretation'] = 'Lower is better' if db_score > 0 else 'Invalid score'
                except Exception as e:
                    self.logger.warning(f"⚠️ Davies-Bouldin score calculation failed: {e}")
                    metrics['davies_bouldin_score'] = None

            # Overall validation score
            valid_scores = [v for v in metrics.values() if v is not None and isinstance(v, (int, float))]
            if valid_scores:
                # Normalize and combine scores (simplified)
                normalized_scores = []
                for score_name, score_value in metrics.items():
                    if score_value is not None and isinstance(score_value, (int, float)):
                        if 'silhouette' in score_name:
                            normalized_scores.append(max(0, min(1, score_value)))
                        elif 'calinski' in score_name:
                            # Normalize CH score (log scale)
                            normalized_scores.append(min(1, score_value / 1000))
                        elif 'davies' in score_name:
                            # Invert DB score (lower is better)
                            normalized_scores.append(max(0, 1 - min(1, score_value)))

                if normalized_scores:
                    metrics['overall_score'] = float(np.mean(normalized_scores))
                    metrics['validation_quality'] = self._interpret_overall_score(metrics['overall_score'])

            validation['metrics'] = metrics
            validation['quality_assessment'] = self._assess_clustering_quality(metrics)

            return validation

        except Exception as e:
            self.logger.warning(f"⚠️ Clustering validation failed: {e}")
            return {'error': str(e)}

    def _interpret_silhouette_score(self, score: float) -> str:
        """
        Interpret silhouette score.

        Args:
            score: Silhouette score

        Returns:
            Interpretation string
        """
        try:
            if score >= 0.7:
                return 'Strong cluster structure'
            elif score >= 0.5:
                return 'Reasonable cluster structure'
            elif score >= 0.25:
                return 'Weak cluster structure'
            else:
                return 'No substantial cluster structure'
        except:
            return 'Unable to interpret'

    def _interpret_overall_score(self, score: float) -> str:
        """
        Interpret overall clustering score.

        Args:
            score: Overall score (0-1)

        Returns:
            Interpretation string
        """
        try:
            if score >= 0.8:
                return 'Excellent clustering quality'
            elif score >= 0.6:
                return 'Good clustering quality'
            elif score >= 0.4:
                return 'Fair clustering quality'
            elif score >= 0.2:
                return 'Poor clustering quality'
            else:
                return 'Very poor clustering quality'
        except:
            return 'Unable to interpret'

    def _assess_clustering_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall clustering quality.

        Args:
            metrics: Validation metrics

        Returns:
            Quality assessment
        """
        try:
            assessment = {'overall_quality': 'Unknown'}

            # Collect valid scores
            scores = {}
            for metric_name, value in metrics.items():
                if value is not None and isinstance(value, (int, float)):
                    scores[metric_name] = value

            if not scores:
                return assessment

            # Simple quality assessment based on available metrics
            if 'overall_score' in scores:
                overall_score = scores['overall_score']
                if overall_score >= 0.7:
                    assessment['overall_quality'] = 'High'
                    assessment['recommendation'] = 'Clustering results are reliable'
                elif overall_score >= 0.5:
                    assessment['overall_quality'] = 'Medium'
                    assessment['recommendation'] = 'Clustering results are acceptable'
                elif overall_score >= 0.3:
                    assessment['overall_quality'] = 'Low'
                    assessment['recommendation'] = 'Consider reviewing clustering parameters'
                else:
                    assessment['overall_quality'] = 'Very Low'
                    assessment['recommendation'] = 'Clustering may not be meaningful'

            # Check for specific issues
            issues = []

            if 'silhouette_score' in scores and scores['silhouette_score'] < 0.25:
                issues.append('Low silhouette score indicates poor cluster separation')

            if 'davies_bouldin_score' in scores and scores['davies_bouldin_score'] > 1.0:
                issues.append('High Davies-Bouldin score indicates overlapping clusters')

            assessment['issues'] = issues
            assessment['confidence_level'] = len(scores) / len(self.config.validation_metrics)

            return assessment

        except Exception as e:
            self.logger.warning(f"⚠️ Quality assessment failed: {e}")
            return {'overall_quality': 'Unknown', 'error': str(e)}

    def _analyze_cluster_characteristics(self,
                                       data: pd.DataFrame,
                                       cluster_labels: np.ndarray,
                                       feature_names: Optional[List[str]] = None) -> Dict[str, ClusterCharacteristics]:
        """
        Analyze detailed characteristics of each cluster.

        Args:
            data: Data to analyze
            cluster_labels: Cluster labels
            feature_names: Feature names

        Returns:
            Dictionary of cluster characteristics
        """
        try:
            characteristics = {}
            unique_clusters = np.unique(cluster_labels)

            for cluster_id in unique_clusters:
                try:
                    # Get cluster data
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = data[cluster_mask]

                    if len(cluster_data) < 5:  # Minimum samples for analysis
                        continue

                    # Calculate characteristics
                    centroid = cluster_data.mean().values
                    variance = cluster_data.var().mean()
                    size = len(cluster_data)

                    # Calculate silhouette score for this cluster
                    try:
                        X_scaled = StandardScaler().fit_transform(data)
                        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                        cluster_silhouette = silhouette_avg  # Simplified
                    except:
                        cluster_silhouette = 0.0

                    # Identify dominant features
                    feature_importance = self._calculate_feature_importance(cluster_data, cluster_id, unique_clusters)
                    dominant_features = [f for f, imp in feature_importance[:3]]  # Top 3 features

                    # Determine market regime
                    market_regime = self._determine_market_regime(cluster_data, cluster_id)

                    # Calculate risk profile
                    risk_profile = self._calculate_risk_profile(cluster_data)

                    # Calculate stability score
                    stability_score = self._calculate_cluster_stability(cluster_data, data, cluster_id, cluster_labels)

                    char = ClusterCharacteristics(
                        cluster_id=cluster_id,
                        size=size,
                        centroid=centroid,
                        variance=variance,
                        silhouette_score=cluster_silhouette,
                        stability_score=stability_score,
                        dominant_features=dominant_features,
                        market_regime=market_regime,
                        risk_profile=risk_profile
                    )

                    characteristics[f'cluster_{cluster_id}'] = char

                except Exception as e:
                    self.logger.warning(f"⚠️ Cluster {cluster_id} characteristics analysis failed: {e}")

            self.logger.info(f"✅ Analyzed characteristics for {len(characteristics)} clusters")
            return characteristics

        except Exception as e:
            self.logger.error(f"❌ Cluster characteristics analysis failed: {e}")
            return {}

    def _calculate_feature_importance(self,
                                    cluster_data: pd.DataFrame,
                                    cluster_id: int,
                                    all_clusters: np.ndarray) -> List[Tuple[str, float]]:
        """
        Calculate feature importance for a cluster.

        Args:
            cluster_data: Data for this cluster
            cluster_id: Cluster ID
            all_clusters: All cluster IDs

        Returns:
            List of (feature_name, importance) tuples
        """
        try:
            # Simplified feature importance calculation
            # In practice, you would use more sophisticated methods
            importance_scores = []

            for col in cluster_data.columns:
                # Calculate how distinctive this feature is for this cluster
                cluster_mean = cluster_data[col].mean()
                cluster_std = cluster_data[col].std()

                # Compare to other clusters (simplified)
                distinctiveness = abs(cluster_mean) / (cluster_std + 1e-8)
                importance_scores.append((col, distinctiveness))

            # Sort by importance
            importance_scores.sort(key=lambda x: x[1], reverse=True)

            return importance_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance calculation failed: {e}")
            return [(col, 0.0) for col in cluster_data.columns]

    def _determine_market_regime(self, cluster_data: pd.DataFrame, cluster_id: int) -> str:
        """
        Determine market regime type for a cluster.

        Args:
            cluster_data: Data for this cluster
            cluster_id: Cluster ID

        Returns:
            Market regime description
        """
        try:
            # This is a simplified regime determination
            # In practice, you would use more sophisticated analysis

            # Analyze cluster characteristics
            volatility = cluster_data.var().mean()
            trend_strength = abs(cluster_data.mean().mean())

            if volatility > cluster_data.var().quantile(0.75):
                if trend_strength > 0.1:
                    return 'High Volatility Trending'
                else:
                    return 'High Volatility Mean Reverting'
            elif volatility < cluster_data.var().quantile(0.25):
                if trend_strength > 0.1:
                    return 'Low Volatility Trending'
                else:
                    return 'Low Volatility Stable'
            else:
                return 'Moderate Volatility Mixed'

        except Exception as e:
            self.logger.warning(f"⚠️ Market regime determination failed: {e}")
            return 'Unknown'

    def _calculate_risk_profile(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk profile for a cluster.

        Args:
            cluster_data: Data for this cluster

        Returns:
            Risk profile dictionary
        """
        try:
            risk_profile = {}

            # Calculate various risk metrics
            if len(cluster_data) > 1:
                # Volatility
                risk_profile['volatility'] = cluster_data.std().mean()

                # Downside risk
                negative_values = (cluster_data < 0).sum().sum()
                risk_profile['downside_exposure'] = negative_values / (cluster_data.size * 100)

                # Concentration risk (simplified)
                max_feature_value = cluster_data.abs().max().max()
                risk_profile['concentration_risk'] = min(max_feature_value, 1.0)

                # Stability
                risk_profile['stability'] = 1.0 / (1.0 + risk_profile['volatility'])

            return risk_profile

        except Exception as e:
            self.logger.warning(f"⚠️ Risk profile calculation failed: {e}")
            return {'volatility': 0.0, 'downside_exposure': 0.0, 'concentration_risk': 0.0, 'stability': 0.0}

    def _calculate_cluster_stability(self,
                                   cluster_data: pd.DataFrame,
                                   all_data: pd.DataFrame,
                                   cluster_id: int,
                                   cluster_labels: np.ndarray) -> float:
        """
        Calculate stability score for a cluster.

        Args:
            cluster_data: Data for this cluster
            all_data: All data
            cluster_id: Cluster ID
            cluster_labels: All cluster labels

        Returns:
            Stability score (0-1)
        """
        try:
            # Simplified stability calculation
            # In practice, you would use bootstrapping or other methods

            # Calculate within-cluster variance
            within_variance = cluster_data.var().mean()

            # Calculate between-cluster distance
            other_clusters_data = all_data[cluster_labels != cluster_id]
            if len(other_clusters_data) > 0:
                cluster_centroid = cluster_data.mean().values
                other_centroid = other_clusters_data.mean().values
                between_distance = np.linalg.norm(cluster_centroid - other_centroid)
            else:
                between_distance = 1.0

            # Stability is higher when within-cluster variance is low and between-cluster distance is high
            if between_distance > 0:
                stability = 1.0 / (1.0 + within_variance / between_distance)
            else:
                stability = 0.0

            stability = max(0.0, min(1.0, stability))

            return stability

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster stability calculation failed: {e}")
            return 0.5

    def _analyze_cluster_stability(self,
                                 data: pd.DataFrame,
                                 cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze cluster stability over time/samples.

        Args:
            data: Data to analyze
            cluster_labels: Cluster labels

        Returns:
            Dictionary of stability analysis
        """
        try:
            analysis = {}

            # Split data into time windows for stability analysis
            n_samples = len(data)
            window_size = max(50, n_samples // 5)  # 5 windows
            windows = []

            for i in range(0, n_samples, window_size):
                window_data = data.iloc[i:i + window_size]
                window_labels = cluster_labels[i:i + window_size]
                if len(window_data) > 10:  # Minimum window size
                    windows.append((window_data, window_labels))

            if len(windows) < 2:
                analysis['stability_score'] = 0.5
                analysis['cluster_drift'] = 0.0
                return analysis

            # Calculate consistency across windows
            consistencies = []
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    consistency = self._calculate_window_consistency(
                        windows[i][0], windows[i][1],
                        windows[j][0], windows[j][1]
                    )
                    consistencies.append(consistency)

            # Overall stability
            if consistencies:
                avg_consistency = np.mean(consistencies)
                stability_score = min(avg_consistency * 1.5, 1.0)  # Scale to 0-1
            else:
                stability_score = 0.5

            analysis['stability_score'] = stability_score
            analysis['window_consistency'] = consistencies
            analysis['n_windows'] = len(windows)
            analysis['window_size'] = window_size
            analysis['cluster_drift'] = 1.0 - stability_score  # Drift is inverse of stability

            return analysis

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster stability analysis failed: {e}")
            return {'stability_score': 0.5, 'error': str(e)}

    def _calculate_window_consistency(self,
                                    data1: pd.DataFrame,
                                    labels1: np.ndarray,
                                    data2: pd.DataFrame,
                                    labels2: np.ndarray) -> float:
        """
        Calculate consistency between two data windows.

        Args:
            data1: First window data
            labels1: First window labels
            data2: Second window data
            labels2: Second window labels

        Returns:
            Consistency score (0-1)
        """
        try:
            # Calculate cluster overlap
            unique_labels1 = set(labels1)
            unique_labels2 = set(labels2)
            overlap = len(unique_labels1 & unique_labels2)
            union = len(unique_labels1 | unique_labels2)

            if union == 0:
                return 0.0

            label_consistency = overlap / union

            # Calculate distribution similarity
            if unique_labels1 and unique_labels2:
                dist1 = pd.Series(labels1).value_counts(normalize=True)
                dist2 = pd.Series(labels2).value_counts(normalize=True)

                # Align distributions
                all_labels = sorted(unique_labels1 | unique_labels2)
                dist1_aligned = np.array([dist1.get(label, 0) for label in all_labels])
                dist2_aligned = np.array([dist2.get(label, 0) for label in all_labels])

                # Jensen-Shannon divergence
                from scipy.spatial.distance import jensenshannon
                js_divergence = jensenshannon(dist1_aligned, dist2_aligned)

                distribution_consistency = 1.0 - js_divergence
            else:
                distribution_consistency = 0.0

            # Combined consistency
            consistency = (label_consistency * 0.5 + distribution_consistency * 0.5)
            consistency = max(0.0, min(1.0, consistency))

            return consistency

        except Exception as e:
            self.logger.warning(f"⚠️ Window consistency calculation failed: {e}")
            return 0.0

    def _generate_clustering_summary(self,
                                   cluster_stats: Dict[str, Any],
                                   validation_results: Dict[str, Any],
                                   cluster_characteristics: Dict[str, ClusterCharacteristics]) -> Dict[str, Any]:
        """
        Generate clustering analysis summary.

        Args:
            cluster_stats: Basic cluster statistics
            validation_results: Validation results
            cluster_characteristics: Cluster characteristics

        Returns:
            Clustering summary
        """
        try:
            summary = {}

            # Overall assessment
            n_clusters = cluster_stats.get('total_clusters', 0)
            balanced = cluster_stats.get('balanced_clusters', False)
            sufficient = cluster_stats.get('sufficient_samples', False)

            if n_clusters >= 3 and balanced and sufficient:
                summary['overall_assessment'] = 'Good clustering structure'
            elif n_clusters >= 2 and sufficient:
                summary['overall_assessment'] = 'Acceptable clustering structure'
            else:
                summary['overall_assessment'] = 'Poor clustering structure - may need review'

            # Key insights
            summary['key_insights'] = []

            if n_clusters < 3:
                summary['key_insights'].append('Consider increasing number of clusters')
            elif n_clusters > 10:
                summary['key_insights'].append('Large number of clusters - consider hierarchical clustering')

            if not balanced:
                summary['key_insights'].append('Clusters are unbalanced - may indicate need for different clustering approach')

            if not sufficient:
                summary['key_insights'].append('Some clusters have insufficient samples')

            # Validation quality
            if 'quality_assessment' in validation_results:
                quality = validation_results['quality_assessment'].get('overall_quality', 'Unknown')
                summary['validation_quality'] = quality
                summary['key_insights'].extend(validation_results['quality_assessment'].get('issues', []))

            # Cluster characteristics summary
            if cluster_characteristics:
                regimes = [char.market_regime for char in cluster_characteristics.values()]
                unique_regimes = set(regimes)
                summary['market_regimes_identified'] = list(unique_regimes)
                summary['regime_diversity'] = len(unique_regimes)

                # Risk profiles
                avg_volatility = np.mean([char.risk_profile.get('volatility', 0) for char in cluster_characteristics.values()])
                summary['average_cluster_volatility'] = avg_volatility

                high_risk_clusters = [cid for cid, char in cluster_characteristics.items()
                                    if char.risk_profile.get('volatility', 0) > avg_volatility * 1.5]
                if high_risk_clusters:
                    summary['high_risk_clusters'] = high_risk_clusters

            # Recommendations
            summary['recommendations'] = []

            if n_clusters < self.config.min_clusters:
                summary['recommendations'].append(f'Consider using at least {self.config.min_clusters} clusters')
            elif n_clusters > self.config.max_clusters:
                summary['recommendations'].append(f'Consider using at most {self.config.max_clusters} clusters')

            if not sufficient:
                summary['recommendations'].append('Collect more data or use different clustering parameters')

            if not balanced:
                summary['recommendations'].append('Consider balancing clusters or using density-based clustering')

            return summary

        except Exception as e:
            self.logger.warning(f"⚠️ Clustering summary generation failed: {e}")
            return {'overall_assessment': 'Unable to generate summary', 'error': str(e)}