"""
Feature Clustering Module

This module groups similar features into coherent market dimensions using
various clustering and similarity-based approaches.

Methods:
- Correlation-based clustering
- Mutual information clustering  
- Graph-based community detection
- Ensemble clustering approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict

from src.utils.logger import system_logger


class ClusteringMethod(Enum):
    """Feature clustering methods."""
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    HIERARCHICAL = "hierarchical"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GRAPH_COMMUNITY = "graph_community"
    ENSEMBLE = "ensemble"


@dataclass
class FeatureClusterResult:
    """Result of feature clustering analysis."""
    method: ClusteringMethod
    feature_groups: Dict[str, List[str]]
    cluster_labels: Dict[str, int]
    similarity_matrix: pd.DataFrame
    cluster_quality_metrics: Dict[str, float]
    group_interpretations: Dict[str, str]


class FeatureClusterer:
    """
    Groups features into coherent market dimensions using clustering methods.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('FeatureClusterer')
    
    def cluster_features(self, 
                        features: pd.DataFrame,
                        method: ClusteringMethod = ClusteringMethod.CORRELATION,
                        similarity_threshold: float = 0.7,
                        n_clusters: Optional[int] = None) -> FeatureClusterResult:
        """
        Cluster features into groups using specified method.
        
        Args:
            features: Feature DataFrame
            method: Clustering method to use
            similarity_threshold: Similarity threshold for clustering
            n_clusters: Number of clusters (if applicable)
            
        Returns:
            Feature clustering results
        """
        
        self.logger.info(f"🔗 Clustering features using {method.value}")
        
        if method == ClusteringMethod.CORRELATION:
            return self._cluster_by_correlation(features, similarity_threshold)
        elif method == ClusteringMethod.MUTUAL_INFORMATION:
            return self._cluster_by_mutual_information(features, similarity_threshold)
        elif method == ClusteringMethod.HIERARCHICAL:
            return self._cluster_hierarchical(features, n_clusters or 6)
        elif method == ClusteringMethod.KMEANS:
            return self._cluster_kmeans(features, n_clusters or 6)
        elif method == ClusteringMethod.DBSCAN:
            return self._cluster_dbscan(features, similarity_threshold)
        elif method == ClusteringMethod.GRAPH_COMMUNITY:
            return self._cluster_graph_community(features, similarity_threshold)
        elif method == ClusteringMethod.ENSEMBLE:
            return self._cluster_ensemble(features, similarity_threshold, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _cluster_by_correlation(self, 
                              features: pd.DataFrame, 
                              threshold: float = 0.7) -> FeatureClusterResult:
        """Cluster features by correlation similarity."""
        
        # Calculate correlation matrix
        correlation_matrix = features.corr().abs()
        
        # Create similarity matrix
        similarity_matrix = correlation_matrix.copy()
        
        # Find highly correlated feature groups
        feature_groups = {}
        cluster_labels = {}
        used_features = set()
        group_id = 0
        
        for feature in correlation_matrix.columns:
            if feature in used_features:
                continue
            
            # Find features highly correlated with this one
            highly_correlated = correlation_matrix[feature][correlation_matrix[feature] >= threshold].index.tolist()
            
            if len(highly_correlated) > 1:  # At least 2 features (including itself)
                group_name = f"correlation_group_{group_id}"
                feature_groups[group_name] = highly_correlated
                
                for feat in highly_correlated:
                    cluster_labels[feat] = group_id
                    used_features.add(feat)
                
                group_id += 1
        
        # Assign remaining features to individual groups
        for feature in correlation_matrix.columns:
            if feature not in used_features:
                group_name = f"singleton_{feature}"
                feature_groups[group_name] = [feature]
                cluster_labels[feature] = group_id
                group_id += 1
        
        # Calculate cluster quality metrics
        quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
        
        # Generate group interpretations
        group_interpretations = self._interpret_correlation_groups(feature_groups, correlation_matrix)
        
        return FeatureClusterResult(
            method=ClusteringMethod.CORRELATION,
            feature_groups=feature_groups,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            cluster_quality_metrics=quality_metrics,
            group_interpretations=group_interpretations
        )
    
    def _cluster_by_mutual_information(self, 
                                     features: pd.DataFrame,
                                     threshold: float = 0.3) -> FeatureClusterResult:
        """Cluster features by mutual information."""
        
        # Calculate mutual information matrix
        mi_matrix = pd.DataFrame(index=features.columns, columns=features.columns)
        
        for col1 in features.columns:
            for col2 in features.columns:
                if col1 == col2:
                    mi_matrix.loc[col1, col2] = 1.0
                else:
                    try:
                        # Use one feature to predict another
                        mi_score = mutual_info_regression(
                            features[[col1]].fillna(0), 
                            features[col2].fillna(0)
                        )[0]
                        # Normalize to [0, 1] range
                        mi_matrix.loc[col1, col2] = min(1.0, mi_score)
                    except:
                        mi_matrix.loc[col1, col2] = 0.0
        
        mi_matrix = mi_matrix.astype(float)
        similarity_matrix = mi_matrix.copy()
        
        # Cluster based on mutual information threshold
        feature_groups = {}
        cluster_labels = {}
        used_features = set()
        group_id = 0
        
        for feature in mi_matrix.columns:
            if feature in used_features:
                continue
            
            # Find features with high mutual information
            high_mi_features = mi_matrix[feature][mi_matrix[feature] >= threshold].index.tolist()
            
            if len(high_mi_features) > 1:
                group_name = f"mi_group_{group_id}"
                feature_groups[group_name] = high_mi_features
                
                for feat in high_mi_features:
                    cluster_labels[feat] = group_id
                    used_features.add(feat)
                
                group_id += 1
        
        # Assign remaining features
        for feature in mi_matrix.columns:
            if feature not in used_features:
                group_name = f"singleton_{feature}"
                feature_groups[group_name] = [feature]
                cluster_labels[feature] = group_id
                group_id += 1
        
        quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
        group_interpretations = self._interpret_mi_groups(feature_groups, mi_matrix)
        
        return FeatureClusterResult(
            method=ClusteringMethod.MUTUAL_INFORMATION,
            feature_groups=feature_groups,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            cluster_quality_metrics=quality_metrics,
            group_interpretations=group_interpretations
        )
    
    def _cluster_hierarchical(self, 
                            features: pd.DataFrame,
                            n_clusters: int = 6) -> FeatureClusterResult:
        """Cluster features using hierarchical clustering."""
        
        # Calculate correlation distance matrix
        correlation_matrix = features.corr().abs()
        distance_matrix = 1 - correlation_matrix
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix.values, method='ward')
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Organize results
        feature_groups = defaultdict(list)
        cluster_labels = {}
        
        for i, feature in enumerate(features.columns):
            cluster_id = cluster_assignments[i] - 1  # Convert to 0-based
            group_name = f"hierarchical_cluster_{cluster_id}"
            feature_groups[group_name].append(feature)
            cluster_labels[feature] = cluster_id
        
        feature_groups = dict(feature_groups)
        similarity_matrix = correlation_matrix
        
        quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
        group_interpretations = self._interpret_hierarchical_groups(feature_groups, correlation_matrix)
        
        return FeatureClusterResult(
            method=ClusteringMethod.HIERARCHICAL,
            feature_groups=feature_groups,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            cluster_quality_metrics=quality_metrics,
            group_interpretations=group_interpretations
        )
    
    def _cluster_kmeans(self, 
                       features: pd.DataFrame,
                       n_clusters: int = 6) -> FeatureClusterResult:
        """Cluster features using K-means on feature space."""
        
        # Transpose features to cluster features (not observations)
        feature_vectors = features.T.fillna(0)
        
        # Standardize feature vectors
        scaler = StandardScaler()
        feature_vectors_scaled = scaler.fit_transform(feature_vectors)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(feature_vectors_scaled)
        
        # Organize results
        feature_groups = defaultdict(list)
        cluster_labels = {}
        
        for i, feature in enumerate(features.columns):
            cluster_id = cluster_assignments[i]
            group_name = f"kmeans_cluster_{cluster_id}"
            feature_groups[group_name].append(feature)
            cluster_labels[feature] = cluster_id
        
        feature_groups = dict(feature_groups)
        
        # Calculate similarity matrix based on cluster assignments
        similarity_matrix = self._calculate_cluster_similarity_matrix(features, cluster_labels)
        
        quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
        group_interpretations = self._interpret_kmeans_groups(feature_groups, features)
        
        return FeatureClusterResult(
            method=ClusteringMethod.KMEANS,
            feature_groups=feature_groups,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            cluster_quality_metrics=quality_metrics,
            group_interpretations=group_interpretations
        )
    
    def _cluster_graph_community(self, 
                               features: pd.DataFrame,
                               threshold: float = 0.5) -> FeatureClusterResult:
        """Cluster features using graph community detection."""
        
        try:
            # Calculate correlation matrix
            correlation_matrix = features.corr().abs()
            
            # Create graph from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for feature in features.columns:
                G.add_node(feature)
            
            # Add edges for correlations above threshold
            for i, feature1 in enumerate(features.columns):
                for j, feature2 in enumerate(features.columns):
                    if i < j:  # Avoid duplicate edges
                        correlation = correlation_matrix.loc[feature1, feature2]
                        if correlation >= threshold:
                            G.add_edge(feature1, feature2, weight=correlation)
            
            # Detect communities
            communities = nx.community.greedy_modularity_communities(G)
            
            # Organize results
            feature_groups = {}
            cluster_labels = {}
            
            for i, community in enumerate(communities):
                group_name = f"community_{i}"
                feature_groups[group_name] = list(community)
                
                for feature in community:
                    cluster_labels[feature] = i
            
            # Handle isolated nodes
            group_id = len(communities)
            for feature in features.columns:
                if feature not in cluster_labels:
                    group_name = f"isolated_{feature}"
                    feature_groups[group_name] = [feature]
                    cluster_labels[feature] = group_id
                    group_id += 1
            
            similarity_matrix = correlation_matrix
            quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
            group_interpretations = self._interpret_community_groups(feature_groups, correlation_matrix)
            
            return FeatureClusterResult(
                method=ClusteringMethod.GRAPH_COMMUNITY,
                feature_groups=feature_groups,
                cluster_labels=cluster_labels,
                similarity_matrix=similarity_matrix,
                cluster_quality_metrics=quality_metrics,
                group_interpretations=group_interpretations
            )
            
        except ImportError:
            self.logger.warning("NetworkX not available, falling back to correlation clustering")
            return self._cluster_by_correlation(features, threshold)
    
    def _cluster_ensemble(self, 
                         features: pd.DataFrame,
                         threshold: float = 0.7,
                         n_clusters: Optional[int] = None) -> FeatureClusterResult:
        """Ensemble clustering combining multiple methods."""
        
        # Run multiple clustering methods
        methods_results = []
        
        # Correlation clustering
        corr_result = self._cluster_by_correlation(features, threshold)
        methods_results.append(corr_result)
        
        # Hierarchical clustering
        hier_result = self._cluster_hierarchical(features, n_clusters or 6)
        methods_results.append(hier_result)
        
        # K-means clustering
        kmeans_result = self._cluster_kmeans(features, n_clusters or 6)
        methods_results.append(kmeans_result)
        
        # Create consensus clustering
        feature_groups, cluster_labels = self._create_consensus_clustering(
            features.columns, methods_results
        )
        
        # Use correlation matrix as similarity matrix
        similarity_matrix = features.corr().abs()
        
        quality_metrics = self._calculate_cluster_quality(features, cluster_labels, similarity_matrix)
        group_interpretations = self._interpret_ensemble_groups(feature_groups, methods_results)
        
        return FeatureClusterResult(
            method=ClusteringMethod.ENSEMBLE,
            feature_groups=feature_groups,
            cluster_labels=cluster_labels,
            similarity_matrix=similarity_matrix,
            cluster_quality_metrics=quality_metrics,
            group_interpretations=group_interpretations
        )
    
    def _create_consensus_clustering(self, 
                                   features: List[str],
                                   methods_results: List[FeatureClusterResult]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """Create consensus clustering from multiple methods."""
        
        # Create co-occurrence matrix
        n_features = len(features)
        feature_to_idx = {feat: i for i, feat in enumerate(features)}
        co_occurrence = np.zeros((n_features, n_features))
        
        # Count how often features appear together across methods
        for result in methods_results:
            for group_features in result.feature_groups.values():
                for i, feat1 in enumerate(group_features):
                    for j, feat2 in enumerate(group_features):
                        if feat1 in feature_to_idx and feat2 in feature_to_idx:
                            idx1 = feature_to_idx[feat1]
                            idx2 = feature_to_idx[feat2]
                            co_occurrence[idx1, idx2] += 1
        
        # Normalize by number of methods
        co_occurrence = co_occurrence / len(methods_results)
        
        # Create final clustering based on consensus
        threshold = 0.6  # Features must appear together in 60% of methods
        
        feature_groups = {}
        cluster_labels = {}
        used_features = set()
        group_id = 0
        
        for i, feature in enumerate(features):
            if feature in used_features:
                continue
            
            # Find features that frequently cluster with this one
            similar_features = []
            for j, other_feature in enumerate(features):
                if co_occurrence[i, j] >= threshold:
                    similar_features.append(other_feature)
            
            if len(similar_features) > 1:
                group_name = f"consensus_group_{group_id}"
                feature_groups[group_name] = similar_features
                
                for feat in similar_features:
                    cluster_labels[feat] = group_id
                    used_features.add(feat)
                
                group_id += 1
        
        # Assign remaining features
        for feature in features:
            if feature not in used_features:
                group_name = f"singleton_{feature}"
                feature_groups[group_name] = [feature]
                cluster_labels[feature] = group_id
                group_id += 1
        
        return feature_groups, cluster_labels
    
    def _calculate_cluster_quality(self, 
                                 features: pd.DataFrame,
                                 cluster_labels: Dict[str, int],
                                 similarity_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate cluster quality metrics."""
        
        quality_metrics = {}
        
        # Intra-cluster similarity (higher is better)
        intra_similarities = []
        inter_similarities = []
        
        unique_clusters = set(cluster_labels.values())
        
        for cluster_id in unique_clusters:
            cluster_features = [feat for feat, cid in cluster_labels.items() if cid == cluster_id]
            
            if len(cluster_features) > 1:
                # Intra-cluster similarities
                for i, feat1 in enumerate(cluster_features):
                    for j, feat2 in enumerate(cluster_features):
                        if i < j and feat1 in similarity_matrix.index and feat2 in similarity_matrix.columns:
                            similarity = similarity_matrix.loc[feat1, feat2]
                            intra_similarities.append(similarity)
                
                # Inter-cluster similarities (with other clusters)
                other_features = [feat for feat, cid in cluster_labels.items() if cid != cluster_id]
                for feat1 in cluster_features:
                    for feat2 in other_features:
                        if feat1 in similarity_matrix.index and feat2 in similarity_matrix.columns:
                            similarity = similarity_matrix.loc[feat1, feat2]
                            inter_similarities.append(similarity)
        
        if intra_similarities:
            quality_metrics['avg_intra_similarity'] = float(np.mean(intra_similarities))
        else:
            quality_metrics['avg_intra_similarity'] = 0.0
        
        if inter_similarities:
            quality_metrics['avg_inter_similarity'] = float(np.mean(inter_similarities))
        else:
            quality_metrics['avg_inter_similarity'] = 0.0
        
        # Silhouette-like score
        if intra_similarities and inter_similarities:
            silhouette_score = (quality_metrics['avg_intra_similarity'] - quality_metrics['avg_inter_similarity']) / \
                             max(quality_metrics['avg_intra_similarity'], quality_metrics['avg_inter_similarity'])
            quality_metrics['silhouette_score'] = float(silhouette_score)
        else:
            quality_metrics['silhouette_score'] = 0.0
        
        # Number of clusters
        quality_metrics['n_clusters'] = len(unique_clusters)
        
        # Average cluster size
        cluster_sizes = [sum(1 for cid in cluster_labels.values() if cid == cluster_id) 
                        for cluster_id in unique_clusters]
        quality_metrics['avg_cluster_size'] = float(np.mean(cluster_sizes))
        quality_metrics['cluster_size_std'] = float(np.std(cluster_sizes))
        
        return quality_metrics
    
    def _calculate_cluster_similarity_matrix(self, 
                                           features: pd.DataFrame,
                                           cluster_labels: Dict[str, int]) -> pd.DataFrame:
        """Calculate similarity matrix based on cluster assignments."""
        
        similarity_matrix = pd.DataFrame(
            index=features.columns,
            columns=features.columns,
            data=0.0
        )
        
        for feat1 in features.columns:
            for feat2 in features.columns:
                if feat1 == feat2:
                    similarity_matrix.loc[feat1, feat2] = 1.0
                elif cluster_labels.get(feat1) == cluster_labels.get(feat2):
                    similarity_matrix.loc[feat1, feat2] = 0.8  # High similarity for same cluster
                else:
                    similarity_matrix.loc[feat1, feat2] = 0.1  # Low similarity for different clusters
        
        return similarity_matrix
    
    def _interpret_correlation_groups(self, 
                                    feature_groups: Dict[str, List[str]],
                                    correlation_matrix: pd.DataFrame) -> Dict[str, str]:
        """Interpret correlation-based feature groups."""
        
        interpretations = {}
        
        for group_name, features in feature_groups.items():
            if len(features) == 1:
                interpretations[group_name] = f"Isolated feature: {features[0]}"
            else:
                # Calculate average correlation within group
                correlations = []
                for i, feat1 in enumerate(features):
                    for j, feat2 in enumerate(features):
                        if i < j:
                            correlations.append(correlation_matrix.loc[feat1, feat2])
                
                avg_corr = np.mean(correlations) if correlations else 0
                
                # Try to identify common themes
                feature_names = [feat.lower() for feat in features]
                
                if any('volume' in name for name in feature_names):
                    theme = "Volume-related"
                elif any('volatility' in name or 'vol' in name for name in feature_names):
                    theme = "Volatility-related"
                elif any('momentum' in name or 'mom' in name for name in feature_names):
                    theme = "Momentum-related"
                elif any('ma' in name or 'moving' in name for name in feature_names):
                    theme = "Moving Average-related"
                elif any('rsi' in name or 'stoch' in name for name in feature_names):
                    theme = "Oscillator-related"
                else:
                    theme = "Mixed features"
                
                interpretations[group_name] = f"{theme} (avg corr: {avg_corr:.3f}, {len(features)} features)"
        
        return interpretations
    
    def _interpret_mi_groups(self, 
                           feature_groups: Dict[str, List[str]],
                           mi_matrix: pd.DataFrame) -> Dict[str, str]:
        """Interpret mutual information-based feature groups."""
        
        interpretations = {}
        
        for group_name, features in feature_groups.items():
            if len(features) == 1:
                interpretations[group_name] = f"Isolated feature: {features[0]}"
            else:
                # Calculate average MI within group
                mi_scores = []
                for i, feat1 in enumerate(features):
                    for j, feat2 in enumerate(features):
                        if i < j:
                            mi_scores.append(mi_matrix.loc[feat1, feat2])
                
                avg_mi = np.mean(mi_scores) if mi_scores else 0
                interpretations[group_name] = f"Mutual information group (avg MI: {avg_mi:.3f}, {len(features)} features)"
        
        return interpretations
    
    def _interpret_hierarchical_groups(self, 
                                     feature_groups: Dict[str, List[str]],
                                     correlation_matrix: pd.DataFrame) -> Dict[str, str]:
        """Interpret hierarchical clustering groups."""
        
        interpretations = {}
        
        for group_name, features in feature_groups.items():
            interpretations[group_name] = f"Hierarchical cluster ({len(features)} features)"
        
        return interpretations
    
    def _interpret_kmeans_groups(self, 
                               feature_groups: Dict[str, List[str]],
                               features: pd.DataFrame) -> Dict[str, str]:
        """Interpret K-means clustering groups."""
        
        interpretations = {}
        
        for group_name, group_features in feature_groups.items():
            interpretations[group_name] = f"K-means cluster ({len(group_features)} features)"
        
        return interpretations
    
    def _interpret_community_groups(self, 
                                  feature_groups: Dict[str, List[str]],
                                  correlation_matrix: pd.DataFrame) -> Dict[str, str]:
        """Interpret graph community detection groups."""
        
        interpretations = {}
        
        for group_name, features in feature_groups.items():
            if "isolated" in group_name:
                interpretations[group_name] = f"Isolated feature: {features[0]}"
            else:
                interpretations[group_name] = f"Community cluster ({len(features)} features)"
        
        return interpretations
    
    def _interpret_ensemble_groups(self, 
                                 feature_groups: Dict[str, List[str]],
                                 methods_results: List[FeatureClusterResult]) -> Dict[str, str]:
        """Interpret ensemble clustering groups."""
        
        interpretations = {}
        
        for group_name, features in feature_groups.items():
            if len(features) == 1:
                interpretations[group_name] = f"Consensus singleton: {features[0]}"
            else:
                interpretations[group_name] = f"Consensus cluster ({len(features)} features, {len(methods_results)} methods)"
        
        return interpretations