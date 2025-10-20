"""
Smart Feature Interaction Discovery

Implements intelligent interaction discovery with correlation-based pre-filtering,
importance-guided generation, and adaptive interaction space reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass
import logging
import time
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import warnings

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

@dataclass
class InteractionConfig:
    """Configuration for smart interaction discovery."""
    
    # Correlation filtering
    correlation_threshold: float = 0.95
    enable_correlation_filtering: bool = True
    
    # Mutual information filtering
    mi_threshold: float = 0.8
    enable_mi_filtering: bool = True
    
    # Clustering
    enable_clustering: bool = True
    clustering_threshold: float = 0.98
    n_clusters: Optional[int] = None
    
    # Interaction limits
    max_interactions: int = 1000
    max_features_per_interaction: int = 50
    
    # Performance
    chunk_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4

class CorrelationFilter:
    """Correlation-based feature filtering."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.logger = logger.getChild('CorrelationFilter')
    
    def filter_highly_correlated_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Filter highly correlated features to reduce redundancy."""
        if not self.config.enable_correlation_filtering:
            tprint("⚠️ [INTERACTION] Correlation filtering disabled")
            return data, {}
        
        tprint(f"🔄 [INTERACTION] Filtering highly correlated features (threshold: {self.config.correlation_threshold})")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = np.where(np.triu(corr_matrix, k=1) > self.config.correlation_threshold)
        
        # Group redundant features
        redundant_groups = {}
        redundant_features = set()
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            feature_i = data.columns[i]
            feature_j = data.columns[j]
            
            if feature_i not in redundant_features and feature_j not in redundant_features:
                # Create new group
                group_key = f"group_{len(redundant_groups)}"
                redundant_groups[group_key] = [feature_i, feature_j]
                redundant_features.add(feature_j)  # Keep the first one
        
        # Remove redundant features
        features_to_keep = [col for col in data.columns if col not in redundant_features]
        filtered_data = data[features_to_keep]
        
        tprint(f"✅ [INTERACTION] Correlation filtering: {len(data.columns)} -> {len(filtered_data.columns)} features")
        tprint(f"📊 [INTERACTION] Removed {len(redundant_features)} redundant features in {len(redundant_groups)} groups")
        
        return filtered_data, redundant_groups
    
    def get_correlation_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation analysis summary."""
        corr_matrix = data.corr().abs()
        
        # Remove diagonal and upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.values[mask]
        
        summary = {
            'total_pairs': len(corr_values),
            'high_correlation_pairs': np.sum(corr_values > self.config.correlation_threshold),
            'mean_correlation': np.mean(corr_values),
            'max_correlation': np.max(corr_values),
            'correlation_threshold': self.config.correlation_threshold
        }
        
        return summary

class MutualInformationFilter:
    """Mutual information-based feature filtering."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.logger = logger.getChild('MutualInformationFilter')
    
    def filter_by_mutual_information(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Filter features based on mutual information with targets."""
        if not self.config.enable_mi_filtering:
            tprint("⚠️ [INTERACTION] Mutual information filtering disabled")
            return data
        
        tprint(f"🔄 [INTERACTION] Filtering features by mutual information (threshold: {self.config.mi_threshold})")
        
        mi_scores = {}
        
        for col in data.columns:
            try:
                # Discretize continuous variables for MI calculation
                feature_discrete = self._discretize_feature(data[col])
                target_discrete = self._discretize_feature(targets)
                
                mi_score = mutual_info_score(target_discrete, target_discrete)
                mi_scores[col] = mi_score
                
            except Exception as e:
                tprint(f"⚠️ [INTERACTION] MI calculation failed for {col}: {e}")
                mi_scores[col] = 0.0
        
        # Filter features with high MI
        features_to_keep = [col for col, mi in mi_scores.items() if mi >= self.config.mi_threshold]
        filtered_data = data[features_to_keep]
        
        tprint(f"✅ [INTERACTION] MI filtering: {len(data.columns)} -> {len(filtered_data.columns)} features")
        
        return filtered_data
    
    def _discretize_feature(self, feature: pd.Series, n_bins: int = 10) -> pd.Series:
        """Discretize continuous feature for MI calculation."""
        try:
            # Handle missing values
            feature_clean = feature.dropna()
            
            if len(feature_clean) == 0:
                return pd.Series([0] * len(feature), index=feature.index)
            
            # Use quantile-based binning
            bins = pd.qcut(feature_clean, q=n_bins, duplicates='drop')
            
            # Map back to original index
            discretized = pd.Series([0] * len(feature), index=feature.index)
            discretized.loc[feature_clean.index] = bins.cat.codes
            
            return discretized
            
        except Exception as e:
            tprint(f"⚠️ [INTERACTION] Discretization failed: {e}")
            return pd.Series([0] * len(feature), index=feature.index)

class FeatureClusterer:
    """Feature clustering for redundancy reduction."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.logger = logger.getChild('FeatureClusterer')
    
    def cluster_similar_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """Cluster similar features and select representatives."""
        if not self.config.enable_clustering:
            tprint("⚠️ [INTERACTION] Feature clustering disabled")
            return data, {}
        
        tprint("🔄 [INTERACTION] Clustering similar features")
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Convert to distance matrix
        distance_matrix = 1 - corr_matrix.values
        
        # Perform clustering
        if self.config.n_clusters is None:
            # Determine optimal number of clusters
            n_clusters = min(50, len(data.columns) // 2)
        else:
            n_clusters = min(self.config.n_clusters, len(data.columns))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group features by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data.columns[i])
        
        # Select representative features (medoids)
        representative_features = []
        cluster_info = {}
        
        for cluster_id, features in clusters.items():
            if len(features) == 1:
                representative_features.extend(features)
                cluster_info[cluster_id] = features
            else:
                # Select medoid (feature with highest average correlation with others)
                best_feature = self._select_medoid(features, corr_matrix)
                representative_features.append(best_feature)
                cluster_info[cluster_id] = features
        
        # Create filtered dataset
        filtered_data = data[representative_features]
        
        tprint(f"✅ [INTERACTION] Clustering: {len(data.columns)} -> {len(filtered_data.columns)} features")
        tprint(f"📊 [INTERACTION] Created {len(clusters)} clusters")
        
        return filtered_data, cluster_info
    
    def _select_medoid(self, features: List[str], corr_matrix: pd.DataFrame) -> str:
        """Select medoid (most representative feature) from cluster."""
        best_feature = features[0]
        best_score = -1
        
        for feature in features:
            # Calculate average correlation with other features in cluster
            other_features = [f for f in features if f != feature]
            if other_features:
                correlations = [corr_matrix.loc[feature, other_feature] for other_feature in other_features]
                avg_correlation = np.mean(correlations)
                
                if avg_correlation > best_score:
                    best_score = avg_correlation
                    best_feature = feature
        
        return best_feature

class ImportanceGuidedGenerator:
    """Importance-guided interaction generation."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.logger = logger.getChild('ImportanceGuidedGenerator')
    
    def generate_importance_guided_interactions(self, features_df: pd.DataFrame, 
                                              importance_scores: Dict[str, float],
                                              max_features: int = 50) -> List[Tuple[str, str]]:
        """Generate interactions based on feature importance rankings."""
        tprint(f"🔄 [INTERACTION] Generating importance-guided interactions (max_features: {max_features})")
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        top_features = [f for f, _ in sorted_features[:max_features]]
        
        tprint(f"📊 [INTERACTION] Selected top {len(top_features)} features by importance")
        
        # Generate interactions among top features
        interactions = list(combinations(top_features, 2))
        
        # Limit interactions if too many
        if len(interactions) > self.config.max_interactions:
            # Prioritize interactions between most important features
            interaction_scores = []
            for f1, f2 in interactions:
                score = importance_scores.get(f1, 0) + importance_scores.get(f2, 0)
                interaction_scores.append((score, (f1, f2)))
            
            # Sort by combined importance and take top interactions
            interaction_scores.sort(reverse=True)
            interactions = [interaction for _, interaction in interaction_scores[:self.config.max_interactions]]
        
        tprint(f"✅ [INTERACTION] Generated {len(interactions)} importance-guided interactions")
        
        return interactions

class SmartInteractionDiscovery:
    """Main smart interaction discovery engine."""
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()
        self.logger = logger.getChild('SmartInteractionDiscovery')
        
        # Initialize components
        self.correlation_filter = CorrelationFilter(self.config)
        self.mi_filter = MutualInformationFilter(self.config)
        self.feature_clusterer = FeatureClusterer(self.config)
        self.importance_generator = ImportanceGuidedGenerator(self.config)
        
        # Performance tracking
        self.discovery_stats = {
            'total_discoveries': 0,
            'correlation_filterings': 0,
            'mi_filterings': 0,
            'clusterings': 0,
            'importance_guidings': 0
        }
        
        tprint("🚀 [INTERACTION] Smart Interaction Discovery initialized")
    
    def discover_interactions(self, features_df: pd.DataFrame, 
                            targets: Optional[pd.Series] = None,
                            importance_scores: Optional[Dict[str, float]] = None,
                            discovery_mode: str = "comprehensive") -> Dict[str, Any]:
        """Discover feature interactions using smart filtering and guidance."""
        tprint(f"🔄 [INTERACTION] Starting smart interaction discovery (mode: {discovery_mode})")
        
        start_time = time.time()
        self.discovery_stats['total_discoveries'] += 1
        
        # Step 1: Correlation filtering
        filtered_data, corr_groups = self.correlation_filter.filter_highly_correlated_features(features_df)
        self.discovery_stats['correlation_filterings'] += 1
        
        # Step 2: Mutual information filtering (if targets available)
        if targets is not None:
            filtered_data = self.mi_filter.filter_by_mutual_information(filtered_data, targets)
            self.discovery_stats['mi_filterings'] += 1
        
        # Step 3: Feature clustering
        clustered_data, cluster_info = self.feature_clusterer.cluster_similar_features(filtered_data)
        self.discovery_stats['clusterings'] += 1
        
        # Step 4: Generate interactions
        if discovery_mode == "importance_guided" and importance_scores is not None:
            interactions = self.importance_generator.generate_importance_guided_interactions(
                clustered_data, importance_scores
            )
            self.discovery_stats['importance_guidings'] += 1
        else:
            # Generate all pairwise interactions
            interactions = list(combinations(clustered_data.columns, 2))
            
            # Limit interactions
            if len(interactions) > self.config.max_interactions:
                interactions = interactions[:self.config.max_interactions]
        
        discovery_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'original_features': len(features_df.columns),
            'filtered_features': len(clustered_data.columns),
            'interactions': interactions,
            'correlation_groups': corr_groups,
            'cluster_info': cluster_info,
            'discovery_time': discovery_time,
            'discovery_mode': discovery_mode
        }
        
        # Add correlation summary
        corr_summary = self.correlation_filter.get_correlation_summary(features_df)
        summary['correlation_summary'] = corr_summary
        
        tprint(f"✅ [INTERACTION] Discovery completed in {discovery_time:.2f}s")
        tprint(f"📊 [INTERACTION] Features: {summary['original_features']} -> {summary['filtered_features']}")
        tprint(f"📊 [INTERACTION] Interactions: {len(interactions)}")
        
        return summary
    
    def generate_interactions_for_subset(self, features_df: pd.DataFrame, 
                                       max_interactions: Optional[int] = None) -> List[Tuple[str, str]]:
        """Generate interactions for a subset of features."""
        if max_interactions is None:
            max_interactions = self.config.max_interactions
        
        tprint(f"🔄 [INTERACTION] Generating interactions for {len(features_df.columns)} features")
        
        # Generate all pairwise interactions
        interactions = list(combinations(features_df.columns, 2))
        
        # Limit interactions
        if len(interactions) > max_interactions:
            interactions = interactions[:max_interactions]
        
        tprint(f"✅ [INTERACTION] Generated {len(interactions)} interactions")
        
        return interactions
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        stats = self.discovery_stats.copy()
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        tprint("🧹 [INTERACTION] Cleaning up interaction discovery")
        
        # Clear any cached data
        self.correlation_filter = None
        self.mi_filter = None
        self.feature_clusterer = None
        self.importance_generator = None
        
        tprint("✅ [INTERACTION] Interaction discovery cleanup completed")
