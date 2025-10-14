"""
Market State Clustering

Define coherent market states based on implicit market dimensions, 
balancing within-cluster homogeneity with sufficient sample counts.

Main Components:
- MarketStateClusterer: Main orchestrator for market state discovery
- RegimeDiscoverer: Core clustering algorithms
- OptimalClusterSelector: Data-driven cluster optimization
- ClusterValidator: Comprehensive cluster validation
- SimilarityClusterer: Similarity-based clustering methods

Usage:
    from research.cluster_analysis.clustering import (
        MarketStateClusterer,
        OptimalClusterSelector,
        ClusterValidator
    )
"""

# Import actual implementations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class MarketStateClusterer:
    """Main orchestrator for market state clustering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.discovered_states = {}
    
    def discover_market_states(self, market_dimensions, n_clusters=None):
        """Discover market states from market dimensions."""
        # Combine all dimension features
        if isinstance(market_dimensions, dict):
            # Concatenate all dimension features
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions
        
        # Handle missing values
        all_features_clean = all_features.fillna(method='ffill').fillna(0)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(all_features_clean)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(all_features_clean)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Create state labels series
        state_labels = pd.Series(cluster_labels, index=all_features.index)
        
        # Calculate state probabilities (distance-based)
        distances = kmeans.transform(features_scaled)
        probabilities_array = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
        
        state_probabilities = pd.DataFrame(
            probabilities_array,
            index=all_features.index,
            columns=[f'state_{i}' for i in range(n_clusters)]
        )
        
        # Create cluster profiles
        cluster_profiles = {}
        for i in range(n_clusters):
            state_mask = state_labels == i
            state_features = all_features_clean[state_mask]
            
            cluster_profiles[f'state_{i}'] = {
                'size': state_mask.sum(),
                'frequency': state_mask.mean(),
                'feature_means': state_features.mean().to_dict(),
                'description': f'Market State {i} ({state_mask.sum()} periods, {state_mask.mean():.1%} frequency)'
            }
        
        results = {
            'labels': state_labels,
            'probabilities': state_probabilities,
            'profiles': cluster_profiles,
            'validation': {'n_clusters': n_clusters, 'inertia': kmeans.inertia_}
        }
        
        self.discovered_states = results
        return results
    
    def find_optimal_clusters(self, market_dimensions, k_range=(2, 8)):
        """Find optimal number of clusters using elbow method."""
        if isinstance(market_dimensions, dict):
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions
        
        all_features_clean = all_features.fillna(method='ffill').fillna(0)
        features_scaled = self.scaler.fit_transform(all_features_clean)
        
        inertias = []
        k_values = list(range(k_range[0], k_range[1] + 1))
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection - find largest decrease
        decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        optimal_k = k_values[np.argmax(decreases)]
        
        # Ensure minimum sample size per cluster
        min_samples_per_cluster = 50
        max_k = len(all_features) // min_samples_per_cluster
        optimal_k = min(optimal_k, max_k)
        
        return max(2, optimal_k)  # At least 2 clusters
    
    def validate_clusters(self, market_dimensions, cluster_labels):
        """Validate cluster quality."""
        if isinstance(market_dimensions, dict):
            all_features = pd.concat(market_dimensions.values(), axis=1)
        else:
            all_features = market_dimensions
        
        all_features_clean = all_features.fillna(method='ffill').fillna(0)
        
        # Calculate silhouette score
        try:
            from sklearn.metrics import silhouette_score
            features_scaled = self.scaler.fit_transform(all_features_clean)
            silhouette = silhouette_score(features_scaled, cluster_labels)
        except:
            silhouette = 0.0
        
        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        
        validation_results = {
            'silhouette_score': silhouette,
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'avg_cluster_size': np.mean(cluster_sizes)
        }
        
        return validation_results

# Placeholder classes - to be implemented during migration
class RegimeDiscoverer:
    """Core clustering algorithms."""
    pass

class OptimalClusterSelector:
    """Data-driven cluster optimization."""
    pass

class SimilarityClusterer:
    """Similarity-based clustering methods."""
    pass

class ClusterValidator:
    """Comprehensive cluster validation."""
    pass

# Main exports
__all__ = [
    "MarketStateClusterer",
    "RegimeDiscoverer",
    "OptimalClusterSelector",
    "SimilarityClusterer", 
    "ClusterValidator"
]