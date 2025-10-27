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

# Import tprint utilities for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_structured,
    tprint_data_preview, tprint_data_format, tprint_feature_counts,
    tprint_timer, tprint_logged
)

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
        tprint_info("🚀 Initializing Enhanced HDBSCAN Clustering Integration")
        
        self.min_features = min_features
        self.max_features = max_features
        self.enable_comprehensive_features = enable_comprehensive_features
        self.enable_pca_reduction = enable_pca_reduction
        self.pca_components = pca_components
        self.pca_variance_threshold = pca_variance_threshold
        self.clustering_config = clustering_config or {}
        
        # Log configuration
        tprint_structured({
            "min_features": min_features,
            "max_features": max_features,
            "enable_comprehensive_features": enable_comprehensive_features,
            "enable_pca_reduction": enable_pca_reduction,
            "pca_components": pca_components,
            "pca_variance_threshold": pca_variance_threshold
        }, level="INFO")
        
        # Initialize feature bank integrator
        if self.enable_comprehensive_features:
            tprint_info("🔧 Configuring Feature Bank Integrator for HDBSCAN clustering")
            
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
            
            tprint_structured({
                "feature_weights": config.hdbscan_weights,
                "target_range": (min_features, max_features)
            }, level="INFO")
            
            self.feature_integrator = FeatureBankIntegrator(config)
            tprint_success("✅ Feature Bank Integrator initialized")
        else:
            tprint_warning("⚠️ Comprehensive features disabled - using basic clustering features")
            self.feature_integrator = None
    
    def get_comprehensive_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive features optimized for HDBSCAN clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing comprehensive features and metadata
        """
        tprint_info("🔍 Generating comprehensive clustering features")
        tprint_data_preview(data, "Input Market Data", max_rows=3, max_cols=8)
        
        with tprint_timer("Feature Generation", level="PERFORMANCE"):
            if self.enable_comprehensive_features:
                tprint_info("📊 Using comprehensive feature bank integration")
                
                # Use comprehensive feature bank integration
                result = self.feature_integrator.get_comprehensive_features_for_task(
                    'hdbscan_clustering', data
                )
                
                tprint_info(f"✅ Generated {len(result.get('feature_names', []))} features")
                
                # Add clustering-specific metadata
                feature_categories = self._get_feature_category_breakdown(result['features'])
                clustering_readiness = self._assess_clustering_readiness(result['features'])
                
                result.update({
                    'clustering_optimized': True,
                    'comprehensive_features': True,
                    'feature_categories': feature_categories,
                    'clustering_readiness': clustering_readiness
                })
                
                # Log feature breakdown
                tprint_structured({
                    "feature_categories": feature_categories,
                    "clustering_readiness": clustering_readiness
                }, level="INFO")
                
                # Preview generated features
                if result.get('features'):
                    tprint_data_preview(
                        pd.DataFrame(result['features']), 
                        "Generated Features", 
                        max_rows=2, 
                        max_cols=5
                    )
                
                return result
            else:
                tprint_warning("⚠️ Using basic clustering features (comprehensive disabled)")
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
        tprint_info("🔧 Preparing data for HDBSCAN clustering")
        tprint_info(f"📊 Pipeline: Feature Generation → Feature Selection ({self.min_features}-{self.max_features}) → PCA ({self.pca_components}) → HDBSCAN")
        
        with tprint_timer("Data Preparation", level="PERFORMANCE"):
            # Get comprehensive features
            feature_result = self.get_comprehensive_clustering_features(data)
            features = feature_result['features']
            feature_names = feature_result['feature_names']
            
            tprint_info(f"📈 Generated {len(feature_names)} features from {len(features)} feature arrays")
            
            if not features:
                tprint_warning("⚠️ No features generated - returning empty arrays")
                return np.array([]).reshape(len(data), 0), [], feature_result
            
            # Convert to numpy array
            tprint_info("🔄 Converting features to numpy array")
            feature_matrix = np.column_stack([features[name] for name in feature_names])
            tprint_data_format(feature_matrix, "Feature Matrix", check_compatibility=True)
            
            # Handle NaN values
            tprint_info("🧹 Handling NaN values")
            nan_count_before = np.isnan(feature_matrix).sum()
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            nan_count_after = np.isnan(feature_matrix).sum()
            
            if nan_count_before > 0:
                tprint_info(f"✅ Replaced {nan_count_before} NaN values with 0.0")
            
            # Standardize features for clustering
            scaler = None
            if SKLEARN_AVAILABLE:
                tprint_info("📏 Standardizing features for clustering")
                scaler = StandardScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
                tprint_success("✅ Features standardized")
            else:
                tprint_warning("⚠️ Scikit-learn not available - skipping standardization")
            
            # Apply PCA reduction if enabled
            pca = None
            original_shape = feature_matrix.shape
            
            if self.enable_pca_reduction and SKLEARN_AVAILABLE and feature_matrix.shape[1] > self.pca_components:
                tprint_info(f"🔬 Applying PCA reduction: {original_shape[1]} → {self.pca_components} components")
                
                # Determine PCA components based on variance threshold or fixed number
                if self.pca_variance_threshold < 1.0:
                    tprint_info(f"📊 Using variance threshold: {self.pca_variance_threshold}")
                    pca = PCA(n_components=self.pca_variance_threshold)
                else:
                    tprint_info(f"📊 Using fixed components: {self.pca_components}")
                    pca = PCA(n_components=min(self.pca_components, feature_matrix.shape[1]))
                
                # Apply PCA
                feature_matrix = pca.fit_transform(feature_matrix)
                
                # Log PCA results
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                tprint_structured({
                    "original_components": original_shape[1],
                    "pca_components": feature_matrix.shape[1],
                    "explained_variance_ratio": explained_variance.tolist(),
                    "cumulative_variance": cumulative_variance.tolist(),
                    "variance_retained": cumulative_variance[-1]
                }, level="INFO")
                
                # Update feature names for PCA components
                feature_names = [f'pca_component_{i+1}' for i in range(feature_matrix.shape[1])]
                tprint_success(f"✅ PCA reduction completed: {original_shape[1]} → {feature_matrix.shape[1]} components")
            else:
                if not self.enable_pca_reduction:
                    tprint_info("⏭️ PCA reduction disabled")
                elif not SKLEARN_AVAILABLE:
                    tprint_warning("⚠️ Scikit-learn not available - skipping PCA")
                else:
                    tprint_info(f"⏭️ Skipping PCA: {feature_matrix.shape[1]} features ≤ {self.pca_components} components")
            
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
            
            # Final data preview
            tprint_data_preview(feature_matrix, "Final Feature Matrix", max_rows=3, max_cols=8)
            tprint_success(f"✅ Data preparation completed: {original_shape} → {feature_matrix.shape}")
            
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
        tprint_info("🎯 Starting Enhanced HDBSCAN Clustering")
        
        if not HDBSCAN_AVAILABLE:
            tprint_error("❌ HDBSCAN not available. Install with: pip install hdbscan")
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        # Log clustering parameters
        tprint_structured({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "metric": metric
        }, level="INFO")
        
        with tprint_timer("HDBSCAN Clustering", level="PERFORMANCE"):
            # Prepare data
            tprint_info("📊 Preparing data for clustering")
            feature_matrix, feature_names, metadata = self.prepare_data_for_clustering(data)
            
            if feature_matrix.size == 0:
                tprint_error("❌ No features available for clustering")
                raise ValueError("No features available for clustering")
            
            tprint_info(f"📈 Feature matrix shape: {feature_matrix.shape}")
            
            # Perform HDBSCAN clustering
            tprint_info("🔄 Performing HDBSCAN clustering")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric=metric
            )
            
            cluster_labels = clusterer.fit_predict(feature_matrix)
            tprint_success("✅ HDBSCAN clustering completed")
            
            # Calculate clustering metrics
            tprint_info("📊 Calculating clustering metrics")
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            noise_ratio = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            
            # Log clustering results
            tprint_structured({
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
                "total_samples": len(cluster_labels)
            }, level="INFO")
            
            # Preview clustering results
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            tprint_info("📊 Cluster distribution:")
            for label, count in zip(unique_labels, counts):
                if label == -1:
                    tprint_info(f"   Noise: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
                else:
                    tprint_info(f"   Cluster {label}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
            
            # Calculate quality metrics
            tprint_info("🔍 Calculating clustering quality metrics")
            quality_metrics = self._calculate_clustering_quality(feature_matrix, cluster_labels)
            
            tprint_structured(quality_metrics, level="INFO")
            tprint_success(f"🎉 Clustering completed successfully: {n_clusters} clusters, {n_noise} noise points")
        
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
        tprint_debug("🔍 Calculating clustering quality metrics")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Scikit-learn not available - returning basic metrics")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': 0.0}
        
        metrics = {}
        
        # Basic statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        total_samples = len(cluster_labels)
        
        tprint_debug(f"📊 Basic stats: {n_clusters} clusters, {n_noise} noise, {total_samples} total samples")
        
        metrics['basic_stats'] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_samples': total_samples,
            'noise_ratio': n_noise / total_samples if total_samples > 0 else 0
        }
        
        # Clustering quality scores
        if n_clusters > 1 and -1 not in cluster_labels:
            tprint_debug("📊 Calculating quality scores (no noise points)")
            try:
                metrics['silhouette_score'] = silhouette_score(feature_matrix, cluster_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(feature_matrix, cluster_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(feature_matrix, cluster_labels)
                tprint_debug(f"✅ Quality scores calculated: silhouette={metrics['silhouette_score']:.4f}")
            except Exception as e:
                tprint_warning(f"⚠️ Error calculating quality scores: {e}")
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = 0.0
        else:
            tprint_debug("⚠️ Cannot calculate quality scores: insufficient clusters or noise present")
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
        
        tprint_debug(f"📊 Overall quality assessment: {quality} (silhouette={silhouette:.4f})")
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
        tprint_info("🔍 Analyzing cluster characteristics")
        
        cluster_labels = clustering_result['cluster_labels']
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        
        tprint_data_preview(feature_matrix, "Feature Matrix for Analysis", max_rows=2, max_cols=5)
        
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in set(cluster_labels) if c != -1]
        tprint_info(f"📊 Analyzing {len(unique_clusters)} clusters (excluding noise)")
        
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
    tprint_info("🚀 Getting Enhanced HDBSCAN Features (Convenience Function)")
    tprint_structured({
        "min_features": min_features,
        "max_features": max_features,
        "enable_pca_reduction": enable_pca_reduction,
        "pca_components": pca_components
    }, level="INFO")
    
    integrator = EnhancedHDBSCANClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        enable_pca_reduction=enable_pca_reduction,
        pca_components=pca_components
    )
    
    result = integrator.get_comprehensive_clustering_features(data)
    tprint_success(f"✅ Enhanced HDBSCAN features generated: {len(result.get('feature_names', []))} features")
    
    return result


def perform_enhanced_hdbscan_clustering(data: pd.DataFrame, 
                                      min_features: int = 100,
                                      max_features: int = 150,
                                      enable_pca_reduction: bool = True,
                                      pca_components: int = 15,
                                      **kwargs) -> Dict[str, Any]:
    """Perform enhanced HDBSCAN clustering with comprehensive features and PCA optimization."""
    tprint_info("🎯 Performing Enhanced HDBSCAN Clustering (Convenience Function)")
    tprint_structured({
        "min_features": min_features,
        "max_features": max_features,
        "enable_pca_reduction": enable_pca_reduction,
        "pca_components": pca_components,
        "clustering_params": kwargs
    }, level="INFO")
    
    integrator = EnhancedHDBSCANClusteringIntegration(
        min_features=min_features,
        max_features=max_features,
        enable_pca_reduction=enable_pca_reduction,
        pca_components=pca_components
    )
    
    result = integrator.cluster_with_enhanced_hdbscan(data, **kwargs)
    tprint_success(f"✅ Enhanced HDBSCAN clustering completed: {result.get('n_clusters', 0)} clusters")
    
    return result


__all__ = [
    'EnhancedHDBSCANClusteringIntegration',
    'get_enhanced_hdbscan_features',
    'perform_enhanced_hdbscan_clustering'
]