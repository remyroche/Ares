"""
Enhanced HDBSCAN Clustering Integration

This module provides comprehensive HDBSCAN clustering integration that combines
existing feature bank features (volume, trend, volatility, momentum) with
clustering-specific features for optimal density-based clustering.

Target: 50-100 comprehensive features optimized for HDBSCAN clustering
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory,
    get_comprehensive_hdbscan_features
)

# Import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")

# Import clustering algorithms and dimensionality reduction
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")


class EnhancedHDBSCANClusteringIntegration:
    """
    Enhanced HDBSCAN Clustering Integration.
    
    Provides 100-150 comprehensive features optimized for density-based clustering
    by combining existing feature bank features with clustering-specific features,
    then reduces dimensionality using PCA to 10-25 components for optimal performance.
    
    Pipeline: Feature Generation → Feature Selection (100-150) → PCA (10-25) → HDBSCAN
    """
    
    def __init__(self, 
                 min_features: int = 100,
                 max_features: int = 150,
                 enable_comprehensive_features: bool = True,
                 enable_pca_reduction: bool = True,
                 pca_components: int = 15,
                 pca_variance_threshold: float = 0.95,
                 clustering_config: Optional[Dict[str, Any]] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        self.pca_variance_threshold = pca_variance_threshold
        self.clustering_config = clustering_config or {}
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            # Configure for HDBSCAN clustering
            config = FeatureBankConfig()
            config.hdbscan_min_features = min_features
            config.hdbscan_max_features = max_features
            # Weight clustering features more heavily
            config.hdbscan_weights = {
                FeatureBankCategory.CLUSTERING: 0.4,  # Clustering-specific features
                FeatureBankCategory.VOLUME: 0.2,      # Volume patterns and clustering
                FeatureBankCategory.TREND: 0.15,      # Trend patterns
                FeatureBankCategory.VOLATILITY: 0.15, # Volatility clustering
                FeatureBankCategory.MOMENTUM: 0.1     # Momentum patterns
            }
            self.feature_integrator = FeatureBankIntegrator(config)
        else:
            self.feature_integrator = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for HDBSCAN clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        if self.enable_comprehensive_features:
            # Use comprehensive feature bank integration
            result = self.feature_integrator.get_comprehensive_features_for_task(
                'hdbscan_clustering', data
            )
            
            # Add clustering-specific metadata
            result.update({
                'clustering_optimized': True,
                'comprehensive_features': True,
                'feature_categories': self._get_feature_category_breakdown(result['features']),
                'clustering_readiness': self._assess_clustering_readiness(result['features'])
            })
            
            return result
        else:
            # Fallback to basic clustering features
            return self._get_basic_clustering_features(data)
    
    def _get_basic_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic clustering features if comprehensive features are disabled."""
        # This would use the original clustering features only
        # For now, return a basic implementation
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'target_range': (self.min_features, self.max_features),
            'clustering_optimized': True,
            'comprehensive_features': False,
            'description': 'Basic clustering features (comprehensive disabled)'
        }
    
    def _get_feature_category_breakdown(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get breakdown of features by category."""
        breakdown = {
            'volume': 0,
            'trend': 0,
            'volatility': 0,
            'momentum': 0,
            'clustering': 0,
            'regime': 0,
            'other': 0
        }
        
        for feature_name in features.keys():
            if any(keyword in feature_name.lower() for keyword in ['volume', 'obv', 'ad', 'mfi', 'vwap']):
                breakdown['volume'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['trend', 'sma', 'ema', 'adx', 'directional']):
                breakdown['trend'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['volatility', 'bollinger', 'atr', 'vol']):
                breakdown['volatility'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['rsi', 'macd', 'stochastic', 'momentum']):
                breakdown['momentum'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['cluster', 'distance', 'separation', 'stability']):
                breakdown['clustering'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['regime', 'entropy', 'complexity', 'hurst']):
                breakdown['regime'] += 1
            else:
                breakdown['other'] += 1
        
        return breakdown
    
    def _assess_clustering_readiness(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess how well-suited the features are for clustering."""
        if not features:
            return {'score': 0, 'issues': ['No features available']}
        
        issues = []
        score = 100
        
        # Check feature count
        feature_count = len(features)
        if feature_count < self.min_features:
            issues.append(f'Too few features: {feature_count} < {self.min_features}')
            score -= 30
        elif feature_count > self.max_features:
            issues.append(f'Too many features: {feature_count} > {self.max_features}')
            score -= 10
        
        # Check feature quality
        quality_issues = 0
        for name, values in features.items():
            if len(values) == 0:
                quality_issues += 1
            elif np.all(np.isnan(values)):
                quality_issues += 1
            elif np.all(values == values[0]):  # All same value
                quality_issues += 1
        
        if quality_issues > 0:
            issues.append(f'{quality_issues} features have quality issues')
            score -= quality_issues * 5
        
        # Check feature diversity
        category_breakdown = self._get_feature_category_breakdown(features)
        unique_categories = sum(1 for count in category_breakdown.values() if count > 0)
        if unique_categories < 3:
            issues.append(f'Low feature diversity: only {unique_categories} categories')
            score -= 20
        
        return {
            'score': max(0, score),
            'issues': issues,
            'feature_count': feature_count,
            'category_diversity': unique_categories,
            'quality_issues': quality_issues
        }
    
    def prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for HDBSCAN clustering with comprehensive features and PCA reduction.
        
        Pipeline: Feature Generation → Feature Selection (100-150) → PCA (10-25) → HDBSCAN
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names, metadata)
        """
        # Get comprehensive features
        feature_result = self.get_comprehensive_clustering_features(data)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        if not features:
            # Return empty arrays if no features
            return np.array([]).reshape(len(data), 0), [], feature_result
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize features for clustering
        scaler = None
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        
        # Apply PCA reduction if enabled
        pca = None
        original_shape = feature_matrix.shape
        if self.enable_pca_reduction and SKLEARN_AVAILABLE and feature_matrix.shape[1] > self.pca_components:
            # Determine PCA components based on variance threshold or fixed number
            if self.pca_variance_threshold < 1.0:
                # Use variance threshold
                pca = PCA(n_components=self.pca_variance_threshold)
            else:
                # Use fixed number of components
                pca = PCA(n_components=min(self.pca_components, feature_matrix.shape[1]))
            
            # Apply PCA
            feature_matrix = pca.fit_transform(feature_matrix)
            
            # Update feature names for PCA components
            feature_names = [f'pca_component_{i+1}' for i in range(feature_matrix.shape[1])]
        
        # Add preprocessing metadata
        metadata = feature_result.copy()
        metadata.update({
            'preprocessing': {
                'scaled': SKLEARN_AVAILABLE,
                'nan_handled': True,
                'original_shape': original_shape,
                'final_shape': feature_matrix.shape,
                'pca_applied': pca is not None,
                'pca_components': feature_matrix.shape[1] if pca is not None else original_shape[1],
                'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist() if pca is not None else None,
                'pca_cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist() if pca is not None else None
            }
        })
        
        return feature_matrix, feature_names, metadata
    
    def cluster_with_enhanced_hdbscan(self, data: pd.DataFrame, 
                                    min_cluster_size: int = 5,
                                    min_samples: int = 3,
                                    cluster_selection_epsilon: float = 0.0,
                                    metric: str = 'euclidean') -> Dict[str, Any]:
        """
        Perform enhanced HDBSCAN clustering with comprehensive features.
        
        Args:
            data: Market data DataFrame
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            cluster_selection_epsilon: Cluster selection epsilon for HDBSCAN
            metric: Distance metric for HDBSCAN
            
        Returns:
            Dictionary containing clustering results
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        # Prepare data
        feature_matrix, feature_names, metadata = self.prepare_data_for_clustering(data)
        
        if feature_matrix.size == 0:
            raise ValueError("No features available for clustering")
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Calculate clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_clustering_quality(feature_matrix, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'clusterer': clusterer,
            'metadata': metadata,
            'quality_metrics': quality_metrics,
            'clustering_parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': cluster_selection_epsilon,
                'metric': metric
            }
        }
    
    def _calculate_clustering_quality(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive clustering quality metrics."""
        if not SKLEARN_AVAILABLE:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': 0.0}
        
        metrics = {}
        
        # Basic statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        total_samples = len(cluster_labels)
        
        metrics['basic_stats'] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_samples': total_samples,
            'noise_ratio': n_noise / total_samples if total_samples > 0 else 0
        }
        
        # Clustering quality scores
        if n_clusters > 1 and -1 not in cluster_labels:
            try:
                metrics['silhouette_score'] = silhouette_score(feature_matrix, cluster_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(feature_matrix, cluster_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(feature_matrix, cluster_labels)
            except:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = 0.0
        
        # Overall quality assessment
        silhouette = metrics['silhouette_score']
        if silhouette > 0.5:
            quality = 'excellent'
        elif silhouette > 0.3:
            quality = 'good'
        elif silhouette > 0.1:
            quality = 'fair'
        else:
            quality = 'poor'
        
        metrics['overall_quality'] = quality
        
        return metrics
    
    def analyze_cluster_characteristics(self, data: pd.DataFrame, 
                                      clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of each cluster using comprehensive features.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_with_enhanced_hdbscan
            
        Returns:
            Dictionary containing cluster analysis
        """
        cluster_labels = clustering_result['cluster_labels']
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        analysis = {
            'n_clusters': len(unique_clusters),
            'cluster_characteristics': {},
            'feature_importance_by_cluster': {},
            'market_characteristics': {}
        }
        
        # Analyze each cluster
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_features = feature_matrix[cluster_mask]
            
            # Basic cluster info
            cluster_info = {
                'size': np.sum(cluster_mask),
                'percentage': np.sum(cluster_mask) / len(cluster_labels) * 100,
                'feature_means': {},
                'feature_stds': {}
            }
            
            # Feature statistics
            for i, feature_name in enumerate(feature_names):
                feature_values = cluster_features[:, i]
                cluster_info['feature_means'][feature_name] = float(np.mean(feature_values))
                cluster_info['feature_stds'][feature_name] = float(np.std(feature_values))
            
            # Market characteristics
            if 'close' in cluster_data.columns:
                prices = cluster_data['close']
                returns = prices.pct_change().dropna()
                
                cluster_info['market_characteristics'] = {
                    'avg_return': float(returns.mean()),
                    'volatility': float(returns.std()),
                    'price_range': float(prices.max() - prices.min()),
                    'trend_direction': 'up' if prices.iloc[-1] > prices.iloc[0] else 'down'
                }
            
            analysis['cluster_characteristics'][f'cluster_{cluster_id}'] = cluster_info
        
        return analysis
    
    def get_feature_importance_for_clustering(self, data: pd.DataFrame, 
                                            clustering_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance for clustering using comprehensive features.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_with_enhanced_hdbscan
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        cluster_labels = clustering_result['cluster_labels']
        
        # Calculate feature importance based on cluster separation
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = feature_matrix[:, i]
            
            # Calculate variance within each cluster
            cluster_variances = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_values = feature_values[cluster_mask]
                
                if len(cluster_values) > 1:
                    cluster_var = np.var(cluster_values)
                    cluster_variances.append(cluster_var)
            
            # Feature importance is inverse of average cluster variance
            if cluster_variances:
                avg_cluster_variance = np.mean(cluster_variances)
                importance_scores[feature_name] = 1 / (avg_cluster_variance + 1e-8)
            else:
                importance_scores[feature_name] = 0.0
        
        return importance_scores


# Convenience functions
def get_enhanced_hdbscan_features(data: pd.DataFrame, 
                                min_features: int = 100,
                                max_features: int = 150,
                                enable_pca_reduction: bool = True,
                                pca_components: int = 15) -> Dict[str, Any]:
    """Get enhanced comprehensive features for HDBSCAN clustering with PCA optimization."""
    integrator = EnhancedHDBSCANClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        enable_pca_reduction=enable_pca_reduction,
        pca_components=pca_components
    )
    return integrator.get_comprehensive_clustering_features(data)


def perform_enhanced_hdbscan_clustering(data: pd.DataFrame, 
                                      min_features: int = 100,
                                      max_features: int = 150,
                                      enable_pca_reduction: bool = True,
                                      pca_components: int = 15,
                                      **kwargs) -> Dict[str, Any]:
    """Perform enhanced HDBSCAN clustering with comprehensive features and PCA optimization."""
    integrator = EnhancedHDBSCANClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        enable_pca_reduction=enable_pca_reduction,
        pca_components=pca_components
    )
    return integrator.cluster_with_enhanced_hdbscan(data, **kwargs)


__all__ = [
    'EnhancedHDBSCANClusteringIntegration',
    'get_enhanced_hdbscan_features',
    'perform_enhanced_hdbscan_clustering'
]