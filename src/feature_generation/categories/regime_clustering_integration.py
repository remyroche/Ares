"""
Regime Clustering Integration

This module provides integration between regime clustering features and the regime clustering task.
It ensures 40-80 features are properly selected for general regime identification and clustering.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Import feature categorization
from .regime_feature_categorization import FeatureUseCase, get_regime_clustering_features
from .regime_features import RegimeFeatureIntegration
from .feature_task_integration import FeatureTaskIntegrator, MLTask

# Import clustering algorithms
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")


class RegimeClusteringIntegration:
    """
    Regime Clustering Integration.
    
    Provides 40-80 features optimized for general regime identification and clustering.
    """
    
    def __init__(self, 
                 min_features: int = 40,
                 max_features: int = 80,
                 clustering_algorithm: str = 'kmeans'):
        self.min_features = min_features
        self.max_features = max_features
        self.clustering_algorithm = clustering_algorithm
        
        # Initialize feature integrator
        self.feature_integrator = FeatureTaskIntegrator()
        
        # Initialize regime feature generator
        self.regime_generator = RegimeFeatureIntegration()
    
    def get_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get features optimized for regime clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing features and metadata
        """
        # Get features from the task integrator
        feature_result = self.feature_integrator.get_features_for_task(
            MLTask.REGIME_CLUSTERING, data
        )
        
        # Generate actual regime features
        regime_features = self.regime_generator.generate_features(data)
        
        # Ensure we have the right number of features
        feature_names = list(regime_features.keys())
        if len(feature_names) > self.max_features:
            # Select top features by regime relevance
            feature_scores = self._score_features_for_regime_relevance(data, regime_features)
            
            # Sort by relevance score and select top features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in sorted_features[:self.max_features]]
            
            # Filter regime features
            filtered_features = {name: regime_features[name] for name in selected_features}
            regime_features = filtered_features
            feature_names = selected_features
        
        # Ensure minimum features
        if len(feature_names) < self.min_features:
            warnings.warn(f"Only {len(feature_names)} features available, minimum is {self.min_features}")
        
        return {
            'features': regime_features,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'target_range': (self.min_features, self.max_features),
            'regime_optimized': True,
            'description': 'Features optimized for regime identification and clustering'
        }
    
    def _score_features_for_regime_relevance(self, data: pd.DataFrame, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Score features for regime relevance."""
        scores = {}
        
        for name, values in features.items():
            # Calculate regime relevance based on:
            # 1. Variance (higher variance = more regime information)
            # 2. Autocorrelation (regime persistence)
            # 3. Non-normality (regime characteristics)
            
            variance_score = np.var(values)
            
            # Autocorrelation (regime persistence)
            if len(values) > 1:
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                autocorr_score = abs(autocorr) if not np.isnan(autocorr) else 0
            else:
                autocorr_score = 0
            
            # Non-normality (regime characteristics)
            if len(values) > 3:
                # Calculate skewness and kurtosis
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val > 0:
                    normalized_values = (values - mean_val) / std_val
                    skewness = np.mean(normalized_values ** 3)
                    kurtosis = np.mean(normalized_values ** 4) - 3
                    non_normality_score = abs(skewness) + abs(kurtosis)
                else:
                    non_normality_score = 0
            else:
                non_normality_score = 0
            
            # Combined score
            scores[name] = variance_score + autocorr_score + non_normality_score
        
        return scores
    
    def prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for regime clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Get regime features
        feature_result = self.get_regime_features(data)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize features
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        return feature_matrix, feature_names
    
    def cluster_regimes(self, data: pd.DataFrame, 
                       n_clusters: Optional[int] = None,
                       algorithm: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform regime clustering on the data.
        
        Args:
            data: Market data DataFrame
            n_clusters: Number of clusters (if None, will be determined automatically)
            algorithm: Clustering algorithm to use
            
        Returns:
            Dictionary containing clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Prepare data
        feature_matrix, feature_names = self.prepare_data_for_clustering(data)
        
        # Determine clustering algorithm
        algorithm = algorithm or self.clustering_algorithm
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(feature_matrix)
        
        # Perform clustering
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        elif algorithm == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Calculate clustering metrics
        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_clustering_quality(feature_matrix, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters_found,
            'n_noise': n_noise,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'clusterer': clusterer,
            'algorithm': algorithm,
            'quality_metrics': quality_metrics
        }
    
    def _determine_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method."""
        if not SKLEARN_AVAILABLE:
            return 3  # Default fallback
        
        max_clusters = min(10, len(feature_matrix) // 10)
        if max_clusters < 2:
            return 2
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            inertias.append(kmeans.inertia_)
            
            if len(set(cluster_labels)) > 1:
                silhouette_scores.append(silhouette_score(feature_matrix, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        # Use silhouette score to determine optimal k
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
            return optimal_k
        else:
            return 3
    
    def _calculate_clustering_quality(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        if not SKLEARN_AVAILABLE:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': 0.0}
        
        metrics = {}
        
        # Silhouette score
        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
            metrics['silhouette_score'] = silhouette_score(feature_matrix, cluster_labels)
        else:
            metrics['silhouette_score'] = 0.0
        
        # Calinski-Harabasz score
        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(feature_matrix, cluster_labels)
        else:
            metrics['calinski_harabasz_score'] = 0.0
        
        # Davies-Bouldin score
        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
            metrics['davies_bouldin_score'] = davies_bouldin_score(feature_matrix, cluster_labels)
        else:
            metrics['davies_bouldin_score'] = 0.0
        
        return metrics
    
    def analyze_regime_characteristics(self, data: pd.DataFrame, 
                                    clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of each regime cluster.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_regimes
            
        Returns:
            Dictionary containing regime analysis
        """
        cluster_labels = clustering_result['cluster_labels']
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        regime_analysis = {
            'n_regimes': len(unique_clusters),
            'regime_characteristics': {},
            'regime_transitions': self._analyze_regime_transitions(cluster_labels),
            'regime_persistence': self._analyze_regime_persistence(cluster_labels)
        }
        
        # Analyze each regime
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_features = feature_matrix[cluster_mask]
            
            regime_analysis['regime_characteristics'][f'regime_{cluster_id}'] = {
                'size': np.sum(cluster_mask),
                'percentage': np.sum(cluster_mask) / len(cluster_labels) * 100,
                'price_stats': self._analyze_price_statistics(cluster_data),
                'feature_stats': self._analyze_feature_statistics(cluster_features, feature_names),
                'volatility_regime': self._classify_volatility_regime(cluster_data),
                'trend_regime': self._classify_trend_regime(cluster_data)
            }
        
        return regime_analysis
    
    def _analyze_regime_transitions(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions."""
        transitions = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                transitions.append((i, cluster_labels[i-1], cluster_labels[i]))
        
        return {
            'total_transitions': len(transitions),
            'transition_rate': len(transitions) / len(cluster_labels),
            'transitions': transitions[:10]  # First 10 transitions
        }
    
    def _analyze_regime_persistence(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime persistence."""
        persistence_lengths = []
        current_regime = cluster_labels[0]
        current_length = 1
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] == current_regime:
                current_length += 1
            else:
                persistence_lengths.append(current_length)
                current_regime = cluster_labels[i]
                current_length = 1
        
        persistence_lengths.append(current_length)
        
        return {
            'avg_persistence': np.mean(persistence_lengths),
            'max_persistence': np.max(persistence_lengths),
            'min_persistence': np.min(persistence_lengths),
            'persistence_std': np.std(persistence_lengths)
        }
    
    def _analyze_price_statistics(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze price statistics for a regime."""
        if 'close' not in cluster_data.columns:
            return {}
        
        prices = cluster_data['close']
        returns = prices.pct_change().dropna()
        
        return {
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'min_price': float(prices.min()),
            'max_price': float(prices.max()),
            'price_range': float(prices.max() - prices.min())
        }
    
    def _analyze_feature_statistics(self, cluster_features: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze feature statistics for a regime."""
        stats = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = cluster_features[:, i]
            stats[feature_name] = {
                'mean': float(np.mean(feature_values)),
                'std': float(np.std(feature_values)),
                'min': float(np.min(feature_values)),
                'max': float(np.max(feature_values))
            }
        
        return stats
    
    def _classify_volatility_regime(self, cluster_data: pd.DataFrame) -> str:
        """Classify volatility regime."""
        if 'close' not in cluster_data.columns:
            return 'unknown'
        
        returns = cluster_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility < 0.01:
            return 'low_volatility'
        elif volatility < 0.03:
            return 'medium_volatility'
        else:
            return 'high_volatility'
    
    def _classify_trend_regime(self, cluster_data: pd.DataFrame) -> str:
        """Classify trend regime."""
        if 'close' not in cluster_data.columns:
            return 'unknown'
        
        prices = cluster_data['close']
        if len(prices) < 2:
            return 'unknown'
        
        price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        if price_change > 0.02:
            return 'uptrend'
        elif price_change < -0.02:
            return 'downtrend'
        else:
            return 'sideways'


# Convenience functions
def get_regime_clustering_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get features for regime clustering."""
    integrator = RegimeClusteringIntegration()
    return integrator.get_regime_features(data)


def perform_regime_clustering(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Perform regime clustering on the data."""
    integrator = RegimeClusteringIntegration()
    return integrator.cluster_regimes(data, **kwargs)


__all__ = [
    'RegimeClusteringIntegration',
    'get_regime_clustering_features',
    'perform_regime_clustering'
]