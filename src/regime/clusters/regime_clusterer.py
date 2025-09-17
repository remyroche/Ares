"""
Regime Clustering Research Module.

This module provides advanced clustering algorithms and techniques specifically
designed for discovering and analyzing market regimes. It implements multiple
clustering approaches to find the most effective regime identification methods.

Key Clustering Approaches:
- Traditional clustering (K-Means, Gaussian Mixture Models)
- Time-series clustering (Dynamic Time Warping, Shape-based)
- Density-based clustering (DBSCAN, HDBSCAN)
- Hierarchical clustering with regime-specific distance metrics
- Ensemble clustering methods
- Online/Streaming clustering for real-time regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
from abc import ABC, abstractmethod

from src.utils.logger import system_logger


class ClusteringMethod(Enum):
    """Enumeration of clustering methods."""
    KMEANS = "kmeans"
    GMM = "gaussian_mixture"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    DTW_CLUSTERING = "dtw_clustering"
    SHAPE_CLUSTERING = "shape_clustering"
    ENSEMBLE = "ensemble"
    ONLINE_KMEANS = "online_kmeans"
    BIRCH = "birch"


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    # General parameters
    n_clusters: int = 5
    random_state: int = 42
    
    # Method-specific parameters
    kmeans_params: Dict[str, Any] = None
    gmm_params: Dict[str, Any] = None
    dbscan_params: Dict[str, Any] = None
    hdbscan_params: Dict[str, Any] = None
    hierarchical_params: Dict[str, Any] = None
    spectral_params: Dict[str, Any] = None
    
    # Time series specific
    dtw_window: Optional[int] = None
    shape_descriptor: str = "slope"  # slope, derivative, fourier
    
    # Ensemble parameters
    ensemble_methods: List[ClusteringMethod] = None
    ensemble_voting: str = "majority"  # majority, weighted, consensus
    
    # Validation parameters
    min_cluster_size: int = 50
    max_clusters: int = 20
    silhouette_threshold: float = 0.3
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.kmeans_params is None:
            self.kmeans_params = {'n_init': 10, 'max_iter': 300}
        if self.gmm_params is None:
            self.gmm_params = {'covariance_type': 'full', 'max_iter': 100}
        if self.dbscan_params is None:
            self.dbscan_params = {'eps': 0.5, 'min_samples': 5}
        if self.hdbscan_params is None:
            self.hdbscan_params = {'min_cluster_size': 50, 'min_samples': 10}
        if self.hierarchical_params is None:
            self.hierarchical_params = {'linkage': 'ward', 'affinity': 'euclidean'}
        if self.spectral_params is None:
            self.spectral_params = {'affinity': 'rbf', 'gamma': 1.0}
        if self.ensemble_methods is None:
            self.ensemble_methods = [ClusteringMethod.KMEANS, ClusteringMethod.GMM, ClusteringMethod.HIERARCHICAL]


@dataclass
class ClusteringResult:
    """Result container for clustering analysis."""
    method: ClusteringMethod
    labels: np.ndarray
    n_clusters: int
    cluster_centers: Optional[np.ndarray]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'labels': self.labels.tolist() if self.labels is not None else None,
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else None,
            'metrics': self.metrics,
            'metadata': self.metadata
        }


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.logger = system_logger.getChild(f'Clusterer.{self.__class__.__name__}')
    
    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        """Fit the clustering algorithm and return predictions."""
        pass
    
    def _calculate_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        metrics = {}
        
        # Only calculate if we have more than one cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(data):
            try:
                metrics['silhouette_score'] = silhouette_score(data, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
            except Exception as e:
                self.logger.warning(f"Could not calculate some metrics: {e}")
        
        # Basic metrics
        metrics['n_clusters'] = len(unique_labels)
        metrics['n_noise'] = np.sum(labels == -1) if -1 in labels else 0
        
        # Cluster size distribution
        cluster_sizes = np.bincount(labels[labels >= 0])
        if len(cluster_sizes) > 0:
            metrics['min_cluster_size'] = int(np.min(cluster_sizes))
            metrics['max_cluster_size'] = int(np.max(cluster_sizes))
            metrics['mean_cluster_size'] = float(np.mean(cluster_sizes))
            metrics['cluster_size_std'] = float(np.std(cluster_sizes))
        
        return metrics


class KMeansClusterer(BaseClusterer):
    """K-Means clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            **self.config.kmeans_params
        )
        
        labels = kmeans.fit_predict(data_scaled)
        metrics = self._calculate_metrics(data_scaled, labels)
        metrics['inertia'] = kmeans.inertia_
        
        return ClusteringResult(
            method=ClusteringMethod.KMEANS,
            labels=labels,
            n_clusters=self.config.n_clusters,
            cluster_centers=scaler.inverse_transform(kmeans.cluster_centers_),
            metrics=metrics,
            metadata={'scaler': scaler, 'model': kmeans}
        )


class GMMClusterer(BaseClusterer):
    """Gaussian Mixture Model clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.config.n_clusters,
            random_state=self.config.random_state,
            **self.config.gmm_params
        )
        
        labels = gmm.fit_predict(data_scaled)
        metrics = self._calculate_metrics(data_scaled, labels)
        metrics['aic'] = gmm.aic(data_scaled)
        metrics['bic'] = gmm.bic(data_scaled)
        metrics['log_likelihood'] = gmm.score(data_scaled)
        
        return ClusteringResult(
            method=ClusteringMethod.GMM,
            labels=labels,
            n_clusters=self.config.n_clusters,
            cluster_centers=scaler.inverse_transform(gmm.means_),
            metrics=metrics,
            metadata={'scaler': scaler, 'model': gmm}
        )


class DBSCANClusterer(BaseClusterer):
    """DBSCAN clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit DBSCAN
        dbscan = DBSCAN(**self.config.dbscan_params)
        labels = dbscan.fit_predict(data_scaled)
        
        # Get cluster centers (mean of each cluster)
        unique_labels = np.unique(labels[labels >= 0])
        cluster_centers = None
        if len(unique_labels) > 0:
            cluster_centers = np.array([
                data_scaled[labels == label].mean(axis=0)
                for label in unique_labels
            ])
            cluster_centers = scaler.inverse_transform(cluster_centers)
        
        metrics = self._calculate_metrics(data_scaled, labels)
        
        return ClusteringResult(
            method=ClusteringMethod.DBSCAN,
            labels=labels,
            n_clusters=len(unique_labels),
            cluster_centers=cluster_centers,
            metrics=metrics,
            metadata={'scaler': scaler, 'model': dbscan}
        )


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        try:
            import hdbscan
        except ImportError:
            self.logger.error("HDBSCAN not available. Install with: pip install hdbscan")
            # Fallback to DBSCAN
            return DBSCANClusterer(self.config).fit_predict(data)
        
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(**self.config.hdbscan_params)
        labels = clusterer.fit_predict(data_scaled)
        
        # Get cluster centers
        unique_labels = np.unique(labels[labels >= 0])
        cluster_centers = None
        if len(unique_labels) > 0:
            cluster_centers = np.array([
                data_scaled[labels == label].mean(axis=0)
                for label in unique_labels
            ])
            cluster_centers = scaler.inverse_transform(cluster_centers)
        
        metrics = self._calculate_metrics(data_scaled, labels)
        
        return ClusteringResult(
            method=ClusteringMethod.HDBSCAN,
            labels=labels,
            n_clusters=len(unique_labels),
            cluster_centers=cluster_centers,
            metrics=metrics,
            metadata={'scaler': scaler, 'model': clusterer}
        )


class HierarchicalClusterer(BaseClusterer):
    """Hierarchical clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            **self.config.hierarchical_params
        )
        
        labels = clustering.fit_predict(data_scaled)
        
        # Get cluster centers
        cluster_centers = np.array([
            data_scaled[labels == label].mean(axis=0)
            for label in range(self.config.n_clusters)
        ])
        cluster_centers = scaler.inverse_transform(cluster_centers)
        
        metrics = self._calculate_metrics(data_scaled, labels)
        
        return ClusteringResult(
            method=ClusteringMethod.HIERARCHICAL,
            labels=labels,
            n_clusters=self.config.n_clusters,
            cluster_centers=cluster_centers,
            metrics=metrics,
            metadata={'scaler': scaler, 'model': clustering}
        )


class SpectralClusterer(BaseClusterer):
    """Spectral clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit Spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            **self.config.spectral_params
        )
        
        labels = clustering.fit_predict(data_scaled)
        
        # Get cluster centers
        cluster_centers = np.array([
            data_scaled[labels == label].mean(axis=0)
            for label in range(self.config.n_clusters)
        ])
        cluster_centers = scaler.inverse_transform(cluster_centers)
        
        metrics = self._calculate_metrics(data_scaled, labels)
        
        return ClusteringResult(
            method=ClusteringMethod.SPECTRAL,
            labels=labels,
            n_clusters=self.config.n_clusters,
            cluster_centers=cluster_centers,
            metrics=metrics,
            metadata={'scaler': scaler, 'model': clustering}
        )


class DTWClusterer(BaseClusterer):
    """Dynamic Time Warping clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        """
        DTW clustering for time series data.
        Note: This is a simplified implementation. For production use,
        consider libraries like tslearn or dtaidistance.
        """
        try:
            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance
            
            # Prepare time series data
            scaler = TimeSeriesScalerMeanVariance()
            data_scaled = scaler.fit_transform(data.reshape(data.shape[0], -1, 1))
            
            # Fit DTW K-Means
            model = TimeSeriesKMeans(
                n_clusters=self.config.n_clusters,
                metric="dtw",
                max_iter=50,
                random_state=self.config.random_state
            )
            
            labels = model.fit_predict(data_scaled)
            
            metrics = self._calculate_metrics(data, labels)
            metrics['inertia'] = model.inertia_
            
            return ClusteringResult(
                method=ClusteringMethod.DTW_CLUSTERING,
                labels=labels,
                n_clusters=self.config.n_clusters,
                cluster_centers=model.cluster_centers_.reshape(self.config.n_clusters, -1),
                metrics=metrics,
                metadata={'scaler': scaler, 'model': model}
            )
            
        except ImportError:
            self.logger.warning("tslearn not available. Falling back to standard K-Means")
            return KMeansClusterer(self.config).fit_predict(data)


class EnsembleClusterer(BaseClusterer):
    """Ensemble clustering implementation."""
    
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        """Combine multiple clustering methods using ensemble voting."""
        
        # Run individual clustering methods
        individual_results = []
        clusterers = {
            ClusteringMethod.KMEANS: KMeansClusterer,
            ClusteringMethod.GMM: GMMClusterer,
            ClusteringMethod.HIERARCHICAL: HierarchicalClusterer,
            ClusteringMethod.SPECTRAL: SpectralClusterer
        }
        
        for method in self.config.ensemble_methods:
            if method in clusterers:
                clusterer = clusterers[method](self.config)
                result = clusterer.fit_predict(data)
                individual_results.append(result)
        
        if not individual_results:
            raise ValueError("No valid clustering methods in ensemble")
        
        # Ensemble voting
        ensemble_labels = self._ensemble_voting(individual_results)
        
        # Calculate ensemble metrics
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        metrics = self._calculate_metrics(data_scaled, ensemble_labels)
        
        # Add ensemble-specific metrics
        metrics['ensemble_agreement'] = self._calculate_agreement(individual_results)
        metrics['ensemble_methods'] = len(individual_results)
        
        # Get cluster centers
        unique_labels = np.unique(ensemble_labels)
        cluster_centers = np.array([
            data[ensemble_labels == label].mean(axis=0)
            for label in unique_labels
        ])
        
        return ClusteringResult(
            method=ClusteringMethod.ENSEMBLE,
            labels=ensemble_labels,
            n_clusters=len(unique_labels),
            cluster_centers=cluster_centers,
            metrics=metrics,
            metadata={'individual_results': individual_results}
        )
    
    def _ensemble_voting(self, results: List[ClusteringResult]) -> np.ndarray:
        """Perform ensemble voting on clustering results."""
        if self.config.ensemble_voting == "majority":
            return self._majority_voting(results)
        elif self.config.ensemble_voting == "weighted":
            return self._weighted_voting(results)
        else:
            return self._consensus_voting(results)
    
    def _majority_voting(self, results: List[ClusteringResult]) -> np.ndarray:
        """Simple majority voting based on cluster co-assignment."""
        n_samples = len(results[0].labels)
        n_methods = len(results)
        
        # Create co-assignment matrix
        co_assignment = np.zeros((n_samples, n_samples))
        
        for result in results:
            labels = result.labels
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_assignment[i, j] += 1
        
        # Threshold for majority
        threshold = n_methods / 2
        co_assignment = (co_assignment > threshold).astype(int)
        
        # Convert to cluster labels using connected components
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(co_assignment)
        
        return labels
    
    def _weighted_voting(self, results: List[ClusteringResult]) -> np.ndarray:
        """Weighted voting based on clustering quality metrics."""
        weights = []
        for result in results:
            # Use silhouette score as weight (default to 0.5 if not available)
            weight = result.metrics.get('silhouette_score', 0.5)
            weights.append(max(0.1, weight))  # Minimum weight of 0.1
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted co-assignment
        n_samples = len(results[0].labels)
        co_assignment = np.zeros((n_samples, n_samples))
        
        for result, weight in zip(results, weights):
            labels = result.labels
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_assignment[i, j] += weight
        
        # Threshold for weighted majority
        threshold = 0.5
        co_assignment = (co_assignment > threshold).astype(int)
        
        # Convert to cluster labels
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(co_assignment)
        
        return labels
    
    def _consensus_voting(self, results: List[ClusteringResult]) -> np.ndarray:
        """Consensus voting requiring agreement from all methods."""
        n_samples = len(results[0].labels)
        n_methods = len(results)
        
        # Create co-assignment matrix
        co_assignment = np.zeros((n_samples, n_samples))
        
        for result in results:
            labels = result.labels
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_assignment[i, j] += 1
        
        # Require consensus from all methods
        co_assignment = (co_assignment == n_methods).astype(int)
        
        # Convert to cluster labels
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(co_assignment)
        
        return labels
    
    def _calculate_agreement(self, results: List[ClusteringResult]) -> float:
        """Calculate agreement between clustering methods."""
        if len(results) < 2:
            return 1.0
        
        from sklearn.metrics import adjusted_rand_score
        
        agreements = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ari = adjusted_rand_score(results[i].labels, results[j].labels)
                agreements.append(ari)
        
        return float(np.mean(agreements))


class RegimeClusterer:
    """
    Main regime clustering research class.
    
    This class provides a comprehensive framework for researching and comparing
    different clustering approaches for market regime identification.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the regime clusterer.
        
        Args:
            config: Configuration for clustering algorithms
        """
        self.config = config or ClusteringConfig()
        self.logger = system_logger.getChild('RegimeClusterer')
        self.results: Dict[ClusteringMethod, ClusteringResult] = {}
        
        # Initialize clusterers
        self.clusterers = {
            ClusteringMethod.KMEANS: KMeansClusterer(self.config),
            ClusteringMethod.GMM: GMMClusterer(self.config),
            ClusteringMethod.DBSCAN: DBSCANClusterer(self.config),
            ClusteringMethod.HDBSCAN: HDBSCANClusterer(self.config),
            ClusteringMethod.HIERARCHICAL: HierarchicalClusterer(self.config),
            ClusteringMethod.SPECTRAL: SpectralClusterer(self.config),
            ClusteringMethod.DTW_CLUSTERING: DTWClusterer(self.config),
            ClusteringMethod.ENSEMBLE: EnsembleClusterer(self.config)
        }
    
    def run_single_method(self, 
                         data: np.ndarray,
                         method: ClusteringMethod) -> ClusteringResult:
        """
        Run a single clustering method.
        
        Args:
            data: Input data for clustering
            method: Clustering method to use
            
        Returns:
            Clustering result
        """
        self.logger.info(f"🔍 Running {method.value} clustering")
        
        if method not in self.clusterers:
            raise ValueError(f"Clustering method {method.value} not supported")
        
        result = self.clusterers[method].fit_predict(data)
        self.results[method] = result
        
        self.logger.info(f"✅ {method.value} completed: {result.n_clusters} clusters, "
                        f"silhouette={result.metrics.get('silhouette_score', 'N/A'):.3f}")
        
        return result
    
    def run_all_methods(self, 
                       data: np.ndarray,
                       analyze_dimensions: bool = True,
                       feature_names: Optional[List[str]] = None) -> Dict[ClusteringMethod, ClusteringResult]:
        """
        Run all available clustering methods with optional dimension analysis.
        
        Args:
            data: Input data for clustering
            analyze_dimensions: Whether to analyze implicit dimensions before clustering
            feature_names: Optional feature names for dimension analysis
            
        Returns:
            Dictionary mapping methods to results
        """
        self.logger.info("🚀 Running comprehensive clustering analysis")
        
        # Analyze implicit dimensions before clustering if requested
        if analyze_dimensions:
            dimension_analysis = self._analyze_implicit_dimensions(data, feature_names)
            self.logger.info(f"📊 Identified {len(dimension_analysis)} implicit dimensions in features")
        
        results = {}
        
        # Run individual methods (exclude ensemble for now)
        individual_methods = [m for m in ClusteringMethod if m != ClusteringMethod.ENSEMBLE]
        
        for method in individual_methods:
            try:
                result = self.run_single_method(data, method)
                
                # Add dimension analysis to result metadata if available
                if analyze_dimensions and 'dimension_analysis' in locals():
                    result.metadata['dimension_analysis'] = dimension_analysis
                
                results[method] = result
            except Exception as e:
                self.logger.error(f"❌ {method.value} failed: {e}")
                continue
        
        # Run ensemble method if we have individual results
        if len(results) >= 2:
            try:
                ensemble_result = self.run_single_method(data, ClusteringMethod.ENSEMBLE)
                if analyze_dimensions and 'dimension_analysis' in locals():
                    ensemble_result.metadata['dimension_analysis'] = dimension_analysis
                results[ClusteringMethod.ENSEMBLE] = ensemble_result
            except Exception as e:
                self.logger.error(f"❌ Ensemble clustering failed: {e}")
        
        self.logger.info(f"✅ Completed {len(results)} clustering methods")
        return results
    
    def _analyze_implicit_dimensions(self, 
                                   data: np.ndarray,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze implicit dimensions in the data before clustering.
        
        Args:
            data: Input data
            feature_names: Optional feature names
            
        Returns:
            Dictionary with dimension analysis results
        """
        self.logger.info("🔍 Analyzing implicit dimensions in feature space")
        
        try:
            from sklearn.decomposition import PCA, FactorAnalysis
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # PCA analysis to identify main dimensions
            pca = PCA()
            pca_transformed = pca.fit_transform(data_scaled)
            
            # Determine number of significant components (cumulative variance > 95%)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumvar >= 0.95) + 1
            n_components_90 = np.argmax(cumvar >= 0.90) + 1
            
            # Factor analysis to identify latent factors
            try:
                fa = FactorAnalysis(n_components=min(10, data.shape[1] // 2))
                fa.fit(data_scaled)
                factor_loadings = fa.components_
            except Exception as e:
                self.logger.warning(f"Factor analysis failed: {e}")
                factor_loadings = None
            
            # Feature importance in principal components
            feature_importance_pc1 = abs(pca.components_[0]) if len(pca.components_) > 0 else None
            feature_importance_pc2 = abs(pca.components_[1]) if len(pca.components_) > 1 else None
            
            # Create feature importance mapping
            if feature_names and feature_importance_pc1 is not None:
                pc1_importance = dict(zip(feature_names, feature_importance_pc1))
                pc2_importance = dict(zip(feature_names, feature_importance_pc2)) if feature_importance_pc2 is not None else {}
            else:
                pc1_importance = {}
                pc2_importance = {}
            
            dimension_analysis = {
                'n_features': data.shape[1],
                'n_samples': data.shape[0],
                'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumvar.tolist(),
                'n_components_90_var': int(n_components_90),
                'n_components_95_var': int(n_components_95),
                'intrinsic_dimensionality_estimate': int(n_components_90),
                'pc1_importance': pc1_importance,
                'pc2_importance': pc2_importance,
                'factor_loadings_available': factor_loadings is not None,
                'data_variance': float(np.var(data_scaled)),
                'feature_correlations': self._calculate_feature_correlations(data_scaled, feature_names)
            }
            
            self.logger.info(f"📊 Estimated intrinsic dimensionality: {n_components_90} components (90% variance)")
            self.logger.info(f"📊 Top PC1 features: {list(pc1_importance.keys())[:5] if pc1_importance else 'N/A'}")
            
            return dimension_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Dimension analysis failed: {e}")
            return {
                'n_features': data.shape[1],
                'n_samples': data.shape[0],
                'analysis_failed': True,
                'error': str(e)
            }
    
    def _calculate_feature_correlations(self, 
                                      data: np.ndarray,
                                      feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate summary statistics about feature correlations."""
        try:
            corr_matrix = np.corrcoef(data.T)
            
            # Remove diagonal elements
            mask = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix[mask]
            
            # Remove NaN values
            correlations = correlations[~np.isnan(correlations)]
            
            if len(correlations) > 0:
                return {
                    'mean_correlation': float(np.mean(np.abs(correlations))),
                    'max_correlation': float(np.max(np.abs(correlations))),
                    'min_correlation': float(np.min(np.abs(correlations))),
                    'high_correlation_pairs': int(np.sum(np.abs(correlations) > 0.8)),
                    'correlation_std': float(np.std(correlations))
                }
            else:
                return {'correlation_analysis_failed': True}
                
        except Exception as e:
            self.logger.warning(f"Feature correlation analysis failed: {e}")
            return {'correlation_analysis_failed': True, 'error': str(e)}
    
    def optimize_cluster_number(self, 
                               data: np.ndarray,
                               method: ClusteringMethod,
                               k_range: Tuple[int, int] = (2, 15)) -> Dict[int, ClusteringResult]:
        """
        Optimize the number of clusters for a given method.
        
        Args:
            data: Input data for clustering
            method: Clustering method to optimize
            k_range: Range of cluster numbers to test
            
        Returns:
            Dictionary mapping cluster numbers to results
        """
        self.logger.info(f"🎯 Optimizing cluster number for {method.value}")
        
        results = {}
        original_n_clusters = self.config.n_clusters
        
        for k in range(k_range[0], k_range[1] + 1):
            self.config.n_clusters = k
            try:
                result = self.run_single_method(data, method)
                results[k] = result
                self.logger.info(f"   k={k}: silhouette={result.metrics.get('silhouette_score', 'N/A'):.3f}")
            except Exception as e:
                self.logger.warning(f"   k={k}: failed ({e})")
                continue
        
        # Restore original config
        self.config.n_clusters = original_n_clusters
        
        # Find optimal k
        if results:
            best_k = max(results.keys(), key=lambda k: results[k].metrics.get('silhouette_score', -1))
            self.logger.info(f"🎯 Optimal k for {method.value}: {best_k}")
        
        return results
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare clustering methods based on various metrics.
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            self.logger.warning("No clustering results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for method, result in self.results.items():
            row = {
                'method': method.value,
                'n_clusters': result.n_clusters,
                'silhouette_score': result.metrics.get('silhouette_score', np.nan),
                'calinski_harabasz_score': result.metrics.get('calinski_harabasz_score', np.nan),
                'davies_bouldin_score': result.metrics.get('davies_bouldin_score', np.nan),
                'min_cluster_size': result.metrics.get('min_cluster_size', np.nan),
                'max_cluster_size': result.metrics.get('max_cluster_size', np.nan),
                'mean_cluster_size': result.metrics.get('mean_cluster_size', np.nan),
                'n_noise': result.metrics.get('n_noise', 0)
            }
            
            # Add method-specific metrics
            if method == ClusteringMethod.GMM:
                row['aic'] = result.metrics.get('aic', np.nan)
                row['bic'] = result.metrics.get('bic', np.nan)
            elif method == ClusteringMethod.ENSEMBLE:
                row['ensemble_agreement'] = result.metrics.get('ensemble_agreement', np.nan)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate rankings
        if len(df) > 1:
            df['silhouette_rank'] = df['silhouette_score'].rank(ascending=False)
            df['ch_rank'] = df['calinski_harabasz_score'].rank(ascending=False)
            df['db_rank'] = df['davies_bouldin_score'].rank(ascending=True)  # Lower is better
            
            # Composite score (equal weights for now)
            df['composite_score'] = (
                df['silhouette_rank'] + 
                df['ch_rank'] + 
                df['db_rank']
            ) / 3
            
            df['overall_rank'] = df['composite_score'].rank(ascending=True)
        
        return df.sort_values('overall_rank') if 'overall_rank' in df.columns else df
    
    def get_best_method(self) -> Optional[Tuple[ClusteringMethod, ClusteringResult]]:
        """
        Get the best clustering method based on composite score.
        
        Returns:
            Tuple of (method, result) for best method, or None if no results
        """
        comparison_df = self.compare_methods()
        
        if comparison_df.empty:
            return None
        
        best_method_name = comparison_df.iloc[0]['method']
        best_method = ClusteringMethod(best_method_name)
        
        return best_method, self.results[best_method]
    
    def save_results(self, filepath: str):
        """Save clustering results to file."""
        results_dict = {
            method.value: result.to_dict() 
            for method, result in self.results.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"💾 Saved clustering results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load clustering results from file."""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        self.results = {}
        for method_name, result_dict in results_dict.items():
            method = ClusteringMethod(method_name)
            
            # Reconstruct ClusteringResult
            result = ClusteringResult(
                method=method,
                labels=np.array(result_dict['labels']) if result_dict['labels'] else None,
                n_clusters=result_dict['n_clusters'],
                cluster_centers=np.array(result_dict['cluster_centers']) if result_dict['cluster_centers'] else None,
                metrics=result_dict['metrics'],
                metadata=result_dict['metadata']
            )
            
            self.results[method] = result
        
        self.logger.info(f"📂 Loaded clustering results from {filepath}")
    
    def generate_clustering_report(self) -> str:
        """Generate a comprehensive clustering analysis report."""
        if not self.results:
            return "No clustering results available. Run clustering analysis first."
        
        report = []
        report.append("# Market Regime Clustering Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        comparison_df = self.compare_methods()
        if not comparison_df.empty:
            report.append("## Method Comparison Summary")
            report.append("")
            
            for _, row in comparison_df.iterrows():
                report.append(f"**{row['method'].upper()}**")
                report.append(f"- Clusters: {row['n_clusters']}")
                report.append(f"- Silhouette Score: {row['silhouette_score']:.3f}")
                if not np.isnan(row.get('calinski_harabasz_score', np.nan)):
                    report.append(f"- Calinski-Harabasz Score: {row['calinski_harabasz_score']:.3f}")
                if not np.isnan(row.get('davies_bouldin_score', np.nan)):
                    report.append(f"- Davies-Bouldin Score: {row['davies_bouldin_score']:.3f}")
                if 'overall_rank' in row:
                    report.append(f"- Overall Rank: {int(row['overall_rank'])}")
                report.append("")
        
        # Best method
        best_method_result = self.get_best_method()
        if best_method_result:
            best_method, best_result = best_method_result
            report.append("## Recommended Method")
            report.append("")
            report.append(f"**{best_method.value.upper()}** is recommended based on composite scoring.")
            report.append(f"- Number of clusters: {best_result.n_clusters}")
            report.append(f"- Silhouette score: {best_result.metrics.get('silhouette_score', 'N/A'):.3f}")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for method, result in self.results.items():
            report.append(f"### {method.value.upper()}")
            report.append(f"- **Clusters discovered**: {result.n_clusters}")
            
            # Cluster size distribution
            if result.labels is not None:
                unique, counts = np.unique(result.labels[result.labels >= 0], return_counts=True)
                report.append(f"- **Cluster sizes**: {dict(zip(unique, counts))}")
            
            # Key metrics
            report.append("**Quality Metrics:**")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    report.append(f"  - {key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report.append(f"  - {key.replace('_', ' ').title()}: {value}")
            
            report.append("")
        
        return "\n".join(report)