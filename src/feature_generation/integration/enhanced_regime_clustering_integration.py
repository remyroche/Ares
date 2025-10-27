"""
Enhanced Regime Clustering Integration

This module provides comprehensive regime clustering integration that combines
existing feature bank features (volume, trend, volatility, momentum) with
regime-specific features for optimal regime identification.

Target: 40-80 comprehensive features optimized for regime clustering
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

# Import feature bank integration
from .feature_bank_integration import (
    FeatureBankIntegrator, FeatureBankConfig, FeatureBankCategory,
    get_comprehensive_regime_clustering_features
)

# Import clustering algorithms
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Install with: pip install scikit-learn")


class EnhancedRegimeClusteringIntegration:
    """
    Enhanced Regime Clustering Integration.
    
    Provides 40-80 comprehensive features optimized for regime identification
    by combining existing feature bank features with regime-specific features.
    """
    
    def __init__(self, 
                 min_features: int = 40,
                 max_features: int = 80,
                 enable_comprehensive_features: bool = True,
                 clustering_config: Optional[Dict[str, Any]] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.clustering_config = clustering_config or {}
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            # Configure for regime clustering
            config = FeatureBankConfig()
            config.regime_clustering_min_features = min_features
            config.regime_clustering_max_features = max_features
            # Weight regime features more heavily
            config.regime_clustering_weights = {
                FeatureBankCategory.REGIME: 0.4,      # Regime-specific features
                FeatureBankCategory.VOLUME: 0.2,      # Volume patterns
                FeatureBankCategory.TREND: 0.2,       # Trend patterns
                FeatureBankCategory.VOLATILITY: 0.15, # Volatility patterns
                FeatureBankCategory.MOMENTUM: 0.05    # Momentum patterns
            }
            self.feature_integrator = FeatureBankIntegrator(config)
        else:
            self.feature_integrator = None
    
    def get_comprehensive_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for regime clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        if self.enable_comprehensive_features:
            # Use comprehensive feature bank integration
            result = self.feature_integrator.get_comprehensive_features_for_task(
                'regime_clustering', data
            )
            
            # Add regime-specific metadata
            result.update({
                'regime_optimized': True,
                'comprehensive_features': True,
                'feature_categories': self._get_feature_category_breakdown(result['features']),
                'regime_readiness': self._assess_regime_readiness(result['features'])
            })
            
            return result
        else:
            # Fallback to basic regime features
            return self._get_basic_regime_features(data)
    
    def _get_basic_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback to basic regime features if comprehensive features are disabled."""
        # This would use the original regime features only
        # For now, return a basic implementation
        return {
            'features': {},
            'feature_names': [],
            'feature_count': 0,
            'target_range': (self.min_features, self.max_features),
            'regime_optimized': True,
            'comprehensive_features': False,
            'description': 'Basic regime features (comprehensive disabled)'
        }
    
    def _get_feature_category_breakdown(self, features: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get breakdown of features by category."""
        breakdown = {
            'regime': 0,
            'volume': 0,
            'trend': 0,
            'volatility': 0,
            'momentum': 0,
            'clustering': 0,
            'other': 0
        }
        
        for feature_name in features.keys():
            if any(keyword in feature_name.lower() for keyword in ['regime', 'entropy', 'complexity', 'hurst', 'fractal', 'memory']):
                breakdown['regime'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['volume', 'obv', 'ad', 'mfi', 'vwap']):
                breakdown['volume'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['trend', 'sma', 'ema', 'adx', 'directional']):
                breakdown['trend'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['volatility', 'bollinger', 'atr', 'vol']):
                breakdown['volatility'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['rsi', 'macd', 'stochastic', 'momentum']):
                breakdown['momentum'] += 1
            elif any(keyword in feature_name.lower() for keyword in ['cluster', 'distance', 'separation', 'stability']):
                breakdown['clustering'] += 1
            else:
                breakdown['other'] += 1
        
        return breakdown
    
    def _assess_regime_readiness(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Assess how well-suited the features are for regime identification."""
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
        
        # Check regime feature presence
        category_breakdown = self._get_feature_category_breakdown(features)
        regime_features = category_breakdown['regime']
        if regime_features < 5:
            issues.append(f'Insufficient regime features: {regime_features} < 5')
            score -= 25
        
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
        unique_categories = sum(1 for count in category_breakdown.values() if count > 0)
        if unique_categories < 4:
            issues.append(f'Low feature diversity: only {unique_categories} categories')
            score -= 20
        
        return {
            'score': max(0, score),
            'issues': issues,
            'feature_count': feature_count,
            'regime_features': regime_features,
            'category_diversity': unique_categories,
            'quality_issues': quality_issues
        }
    
    def prepare_data_for_regime_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare data for regime clustering with comprehensive features.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names, metadata)
        """
        # Get comprehensive features
        feature_result = self.get_comprehensive_regime_features(data)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        if not features:
            # Return empty arrays if no features
            return np.array([]).reshape(len(data), 0), [], feature_result
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Use robust scaling for regime clustering (less sensitive to outliers)
        if SKLEARN_AVAILABLE:
            scaler = RobustScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        
        # Add preprocessing metadata
        metadata = feature_result.copy()
        metadata.update({
            'preprocessing': {
                'scaled': SKLEARN_AVAILABLE,
                'scaling_method': 'robust',
                'nan_handled': True,
                'feature_matrix_shape': feature_matrix.shape
            }
        })
        
        return feature_matrix, feature_names, metadata
    
    def cluster_with_enhanced_regime_clustering(self, data: pd.DataFrame, 
                                              algorithm: str = 'kmeans',
                                              n_clusters: Optional[int] = None,
                                              **kwargs) -> Dict[str, Any]:
        """
        Perform enhanced regime clustering with comprehensive features.
        
        Args:
            data: Market data DataFrame
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'gmm', 'agglomerative')
            n_clusters: Number of clusters (auto-determined if None)
            **kwargs: Additional parameters for clustering algorithm
            
        Returns:
            Dictionary containing clustering results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Prepare data
        feature_matrix, feature_names, metadata = self.prepare_data_for_regime_clustering(data)
        
        if feature_matrix.size == 0:
            raise ValueError("No features available for clustering")
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(feature_matrix, algorithm)
        
        # Perform clustering
        clusterer, cluster_labels = self._perform_clustering(
            feature_matrix, algorithm, n_clusters, **kwargs
        )
        
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
            'metadata': metadata,
            'quality_metrics': quality_metrics,
            'clustering_parameters': {
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                **kwargs
            }
        }
    
    def _determine_optimal_clusters(self, feature_matrix: np.ndarray, algorithm: str) -> int:
        """Determine optimal number of clusters using multiple methods."""
        if algorithm == 'dbscan':
            return 3  # DBSCAN doesn't require n_clusters
        
        # Test different numbers of clusters
        max_clusters = min(10, len(feature_matrix) // 10)
        if max_clusters < 2:
            return 2
        
        silhouette_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            try:
                if algorithm == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                elif algorithm == 'gmm':
                    clusterer = GaussianMixture(n_components=k, random_state=42)
                elif algorithm == 'agglomerative':
                    clusterer = AgglomerativeClustering(n_clusters=k)
                else:
                    continue
                
                cluster_labels = clusterer.fit_predict(feature_matrix)
                
                # Calculate metrics
                if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                    silhouette_scores.append(silhouette_score(feature_matrix, cluster_labels))
                    calinski_scores.append(calinski_harabasz_score(feature_matrix, cluster_labels))
                else:
                    silhouette_scores.append(0)
                    calinski_scores.append(0)
                    
            except:
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        # Find optimal k based on silhouette score
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
            return optimal_k
        else:
            return 3
    
    def _perform_clustering(self, feature_matrix: np.ndarray, algorithm: str, 
                          n_clusters: int, **kwargs) -> Tuple[Any, np.ndarray]:
        """Perform clustering with the specified algorithm."""
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
            cluster_labels = clusterer.fit_predict(feature_matrix)
            
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
            cluster_labels = clusterer.fit_predict(feature_matrix)
            
        elif algorithm == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42, **kwargs)
            cluster_labels = clusterer.fit_predict(feature_matrix)
            
        elif algorithm == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            cluster_labels = clusterer.fit_predict(feature_matrix)
            
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        return clusterer, cluster_labels
    
    def _calculate_clustering_quality(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive clustering quality metrics."""
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
    
    def analyze_regime_characteristics(self, data: pd.DataFrame, 
                                     clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of each regime using comprehensive features.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_with_enhanced_regime_clustering
            
        Returns:
            Dictionary containing regime analysis
        """
        cluster_labels = clustering_result['cluster_labels']
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        
        analysis = {
            'n_regimes': len(unique_clusters),
            'regime_characteristics': {},
            'regime_transitions': self._analyze_regime_transitions(cluster_labels),
            'regime_stability': self._analyze_regime_stability(cluster_labels)
        }
        
        # Analyze each regime
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_features = feature_matrix[cluster_mask]
            
            # Basic regime info
            regime_info = {
                'size': np.sum(cluster_mask),
                'percentage': np.sum(cluster_mask) / len(cluster_labels) * 100,
                'feature_means': {},
                'feature_stds': {},
                'market_characteristics': {}
            }
            
            # Feature statistics
            for i, feature_name in enumerate(feature_names):
                feature_values = cluster_features[:, i]
                regime_info['feature_means'][feature_name] = float(np.mean(feature_values))
                regime_info['feature_stds'][feature_name] = float(np.std(feature_values))
            
            # Market characteristics
            if 'close' in cluster_data.columns:
                prices = cluster_data['close']
                returns = prices.pct_change().dropna()
                
                regime_info['market_characteristics'] = {
                    'avg_return': float(returns.mean()),
                    'volatility': float(returns.std()),
                    'price_range': float(prices.max() - prices.min()),
                    'trend_direction': 'up' if prices.iloc[-1] > prices.iloc[0] else 'down',
                    'max_drawdown': self._calculate_max_drawdown(prices),
                    'sharpe_ratio': self._calculate_sharpe_ratio(returns)
                }
            
            analysis['regime_characteristics'][f'regime_{cluster_id}'] = regime_info
        
        return analysis
    
    def _analyze_regime_transitions(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transitions."""
        transitions = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                transitions.append({
                    'from_regime': cluster_labels[i-1],
                    'to_regime': cluster_labels[i],
                    'transition_point': i
                })
        
        return {
            'total_transitions': len(transitions),
            'transition_rate': len(transitions) / len(cluster_labels),
            'transitions': transitions
        }
    
    def _analyze_regime_stability(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime stability."""
        # Calculate regime durations
        regime_durations = []
        current_regime = cluster_labels[0]
        current_duration = 1
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = cluster_labels[i]
                current_duration = 1
        
        regime_durations.append(current_duration)  # Add last regime
        
        return {
            'avg_duration': np.mean(regime_durations),
            'min_duration': np.min(regime_durations),
            'max_duration': np.max(regime_durations),
            'duration_std': np.std(regime_durations)
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))  # Annualized
    
    def get_feature_importance_for_regime_clustering(self, data: pd.DataFrame, 
                                                   clustering_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance for regime clustering using comprehensive features.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_with_enhanced_regime_clustering
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        cluster_labels = clustering_result['cluster_labels']
        
        # Calculate feature importance based on regime separation
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = feature_matrix[:, i]
            
            # Calculate between-cluster variance vs within-cluster variance
            cluster_means = []
            cluster_sizes = []
            
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_values = feature_values[cluster_mask]
                
                if len(cluster_values) > 0:
                    cluster_means.append(np.mean(cluster_values))
                    cluster_sizes.append(len(cluster_values))
            
            if len(cluster_means) > 1:
                # Between-cluster variance
                overall_mean = np.mean(feature_values)
                between_var = np.sum([size * (mean - overall_mean)**2 for mean, size in zip(cluster_means, cluster_sizes)])
                
                # Within-cluster variance
                within_var = 0
                for cluster_id in set(cluster_labels):
                    if cluster_id == -1:
                        continue
                    cluster_mask = cluster_labels == cluster_id
                    cluster_values = feature_values[cluster_mask]
                    if len(cluster_values) > 0:
                        cluster_mean = np.mean(cluster_values)
                        within_var += np.sum((cluster_values - cluster_mean)**2)
                
                # F-ratio (between-cluster variance / within-cluster variance)
                if within_var > 0:
                    f_ratio = between_var / within_var
                    importance_scores[feature_name] = f_ratio
                else:
                    importance_scores[feature_name] = 0.0
            else:
                importance_scores[feature_name] = 0.0
        
        return importance_scores


# Convenience functions
def get_enhanced_regime_clustering_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get enhanced comprehensive features for regime clustering."""
    integrator = EnhancedRegimeClusteringIntegration()
    return integrator.get_comprehensive_regime_features(data)


def perform_enhanced_regime_clustering(data: pd.DataFrame, algorithm: str = 'kmeans', **kwargs) -> Dict[str, Any]:
    """Perform enhanced regime clustering with comprehensive features."""
    integrator = EnhancedRegimeClusteringIntegration()
    return integrator.cluster_with_enhanced_regime_clustering(data, algorithm, **kwargs)


__all__ = [
    'EnhancedRegimeClusteringIntegration',
    'get_enhanced_regime_clustering_features',
    'perform_enhanced_regime_clustering'
]