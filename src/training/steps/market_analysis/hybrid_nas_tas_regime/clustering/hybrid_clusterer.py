"""
Hybrid Clusterer

Advanced clustering component that combines TAS and NAS inputs to perform
regime detection with economic and financial relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from sklearn.cluster import KMeans, GaussianMixture, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from ..config.hybrid_config import HybridRegimeConfig, ClusteringMethod


class HybridClusterer:
    """
    Hybrid clustering component that combines TAS and NAS inputs.
    
    This component:
    1. Performs clustering based on combined TAS & NAS inputs
    2. Creates coherent regime modeling with economic/financial relevance
    3. Tags existing data with regime information
    4. Replaces hmm_clustering functionality
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """
        Initialize Hybrid Clusterer.
        
        Args:
            config: Hybrid regime configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize clustering components
        self.scaler = StandardScaler()
        self.clusterers = {}
        self.clustering_results = {}
        
        # Initialize clustering methods
        self._initialize_clusterers()
        
        self.logger.info("✅ Hybrid Clusterer initialized")
        self.logger.info(f"🔍 Clustering method: {config.clustering_method.value}")
        self.logger.info(f"📊 Number of regimes: {config.n_regimes}")
        self.logger.info(f"🎯 Clustering metrics: {config.clustering_metrics}")
    
    def _initialize_clusterers(self):
        """Initialize clustering algorithms."""
        try:
            # KMeans
            self.clusterers['kmeans'] = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42,
                n_init=10,
                max_iter=300,
                tol=1e-4
            )
            
            # Gaussian Mixture
            self.clusterers['gaussian_mixture'] = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42,
                n_init=10,
                max_iter=300,
                tol=1e-4
            )
            
            # Hierarchical Clustering
            self.clusterers['hierarchical'] = AgglomerativeClustering(
                n_clusters=self.config.n_regimes,
                linkage='ward'
            )
            
            # DBSCAN
            self.clusterers['dbscan'] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            self.logger.info("✅ Clustering algorithms initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Clustering initialization failed: {e}")
            raise
    
    def cluster(self, 
                data: np.ndarray,
                n_clusters: Optional[int] = None,
                method: Optional[str] = None,
                timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform hybrid clustering on data.
        
        Args:
            data: Data to cluster
            n_clusters: Number of clusters (uses config default if None)
            method: Clustering method (uses config default if None)
            timestamps: Optional timestamps
            
        Returns:
            Dictionary with clustering results
        """
        start_time = time.time()
        self.logger.info("🔍 Starting hybrid clustering")
        
        try:
            # Use provided parameters or config defaults
            n_clusters = n_clusters or self.config.n_regimes
            method = method or self.config.clustering_method.value
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Perform clustering
            if method == "hybrid":
                clustering_results = self._hybrid_clustering(prepared_data, n_clusters)
            else:
                clustering_results = self._single_method_clustering(prepared_data, n_clusters, method)
            
            # Calculate clustering metrics
            metrics = self._calculate_clustering_metrics(prepared_data, clustering_results)
            
            # Generate regime labels
            regime_labels = self._generate_regime_labels(clustering_results['labels'])
            
            # Calculate regime characteristics
            regime_characteristics = self._calculate_regime_characteristics(
                prepared_data, clustering_results, regime_labels
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Hybrid clustering completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Detected {len(set(clustering_results['labels']))} regimes")
            self.logger.info(f"🎯 Silhouette score: {metrics.get('silhouette_score', 0.0):.3f}")
            
            return {
                'success': True,
                'labels': clustering_results['labels'],
                'centers': clustering_results.get('centers'),
                'probabilities': clustering_results.get('probabilities'),
                'regime_labels': regime_labels,
                'regime_characteristics': regime_characteristics,
                'metrics': metrics,
                'method': method,
                'n_clusters': n_clusters,
                'execution_time': execution_time,
                'metadata': {
                    'data_shape': data.shape,
                    'n_samples': len(data),
                    'n_features': data.shape[1] if len(data.shape) > 1 else 1
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hybrid clustering failed: {e}")
            
            return {
                'success': False,
                'labels': np.array([]),
                'centers': None,
                'probabilities': None,
                'regime_labels': [],
                'regime_characteristics': {},
                'metrics': {},
                'method': method,
                'n_clusters': n_clusters,
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for clustering."""
        self.logger.info("📊 Preparing data for clustering")
        
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Standardize data
        prepared_data = self.scaler.fit_transform(data)
        
        return prepared_data
    
    def _hybrid_clustering(self, data: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Perform hybrid clustering using multiple methods."""
        self.logger.info("🔀 Performing hybrid clustering")
        
        # Try multiple clustering methods
        methods = ['kmeans', 'gaussian_mixture', 'hierarchical']
        results = {}
        
        for method in methods:
            try:
                if method == 'kmeans':
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(data)
                    centers = clusterer.cluster_centers_
                    probabilities = None
                    
                elif method == 'gaussian_mixture':
                    clusterer = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(data)
                    centers = clusterer.means_
                    probabilities = clusterer.predict_proba(data)
                    
                elif method == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    labels = clusterer.fit_predict(data)
                    centers = self._calculate_hierarchical_centers(data, labels, n_clusters)
                    probabilities = None
                
                # Calculate metrics for this method
                metrics = self._calculate_clustering_metrics(data, {
                    'labels': labels,
                    'centers': centers,
                    'probabilities': probabilities
                })
                
                results[method] = {
                    'labels': labels,
                    'centers': centers,
                    'probabilities': probabilities,
                    'metrics': metrics
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Clustering method {method} failed: {e}")
                continue
        
        # Select best method based on silhouette score
        best_method = None
        best_score = -1
        
        for method, result in results.items():
            silhouette = result['metrics'].get('silhouette_score', -1)
            if silhouette > best_score:
                best_score = silhouette
                best_method = method
        
        if best_method is None:
            # Fallback to kmeans
            best_method = 'kmeans'
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
            probabilities = None
        else:
            labels = results[best_method]['labels']
            centers = results[best_method]['centers']
            probabilities = results[best_method]['probabilities']
        
        return {
            'labels': labels,
            'centers': centers,
            'probabilities': probabilities,
            'method': best_method,
            'all_results': results
        }
    
    def _single_method_clustering(self, data: np.ndarray, n_clusters: int, method: str) -> Dict[str, Any]:
        """Perform clustering using a single method."""
        self.logger.info(f"🔍 Performing {method} clustering")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
            probabilities = None
            
        elif method == 'gaussian_mixture':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(data)
            centers = clusterer.means_
            probabilities = clusterer.predict_proba(data)
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clusterer.fit_predict(data)
            centers = self._calculate_hierarchical_centers(data, labels, n_clusters)
            probabilities = None
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(data)
            centers = self._calculate_dbscan_centers(data, labels)
            probabilities = None
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return {
            'labels': labels,
            'centers': centers,
            'probabilities': probabilities,
            'method': method
        }
    
    def _calculate_hierarchical_centers(self, data: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """Calculate cluster centers for hierarchical clustering."""
        centers = np.zeros((n_clusters, data.shape[1]))
        
        for i in range(n_clusters):
            cluster_data = data[labels == i]
            if len(cluster_data) > 0:
                centers[i] = np.mean(cluster_data, axis=0)
            else:
                centers[i] = np.mean(data, axis=0)
        
        return centers
    
    def _calculate_dbscan_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers for DBSCAN."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise
        
        centers = np.zeros((n_clusters, data.shape[1]))
        
        for i, label in enumerate(unique_labels):
            if label != -1:  # Skip noise
                cluster_data = data[labels == label]
                if len(cluster_data) > 0:
                    centers[i] = np.mean(cluster_data, axis=0)
                else:
                    centers[i] = np.mean(data, axis=0)
        
        return centers
    
    def _calculate_clustering_metrics(self, data: np.ndarray, clustering_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        labels = clustering_results['labels']
        
        if len(set(labels)) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }
        
        metrics = {}
        
        try:
            # Silhouette score
            if 'silhouette_score' in self.config.clustering_metrics:
                metrics['silhouette_score'] = silhouette_score(data, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Silhouette score calculation failed: {e}")
            metrics['silhouette_score'] = 0.0
        
        try:
            # Calinski-Harabasz score
            if 'calinski_harabasz_score' in self.config.clustering_metrics:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Calinski-Harabasz score calculation failed: {e}")
            metrics['calinski_harabasz_score'] = 0.0
        
        try:
            # Davies-Bouldin score
            if 'davies_bouldin_score' in self.config.clustering_metrics:
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
        except Exception as e:
            self.logger.warning(f"⚠️ Davies-Bouldin score calculation failed: {e}")
            metrics['davies_bouldin_score'] = float('inf')
        
        return metrics
    
    def _generate_regime_labels(self, labels: np.ndarray) -> List[str]:
        """Generate regime labels from cluster labels."""
        unique_labels = np.unique(labels)
        regime_labels = []
        
        for label in unique_labels:
            if label == 0:
                regime_labels.append("normal")
            elif label == 1:
                regime_labels.append("bull_market")
            elif label == 2:
                regime_labels.append("bear_market")
            elif label == 3:
                regime_labels.append("high_volatility")
            elif label == 4:
                regime_labels.append("low_volatility")
            elif label == 5:
                regime_labels.append("trending_up")
            elif label == 6:
                regime_labels.append("trending_down")
            elif label == 7:
                regime_labels.append("mean_reverting")
            elif label == 8:
                regime_labels.append("breakout")
            elif label == 9:
                regime_labels.append("consolidation")
            elif label == 10:
                regime_labels.append("crisis")
            else:
                regime_labels.append("unknown")
        
        return regime_labels
    
    def _calculate_regime_characteristics(self, 
                                          data: np.ndarray, 
                                          clustering_results: Dict[str, Any],
                                          regime_labels: List[str]) -> Dict[str, Dict[str, Any]]:
        """Calculate characteristics for each regime."""
        labels = clustering_results['labels']
        centers = clustering_results.get('centers')
        
        regime_characteristics = {}
        
        for i, regime_label in enumerate(regime_labels):
            regime_data = data[labels == i]
            
            if len(regime_data) > 0:
                characteristics = {
                    'n_samples': len(regime_data),
                    'mean': np.mean(regime_data, axis=0).tolist(),
                    'std': np.std(regime_data, axis=0).tolist(),
                    'min': np.min(regime_data, axis=0).tolist(),
                    'max': np.max(regime_data, axis=0).tolist(),
                    'center': centers[i].tolist() if centers is not None else None
                }
                
                # Calculate regime stability
                if len(regime_data) > 1:
                    characteristics['stability'] = float(1.0 - np.std(regime_data) / (np.mean(regime_data) + 1e-8))
                else:
                    characteristics['stability'] = 1.0
                
                # Calculate regime volatility
                if len(regime_data) > 1:
                    characteristics['volatility'] = float(np.std(regime_data))
                else:
                    characteristics['volatility'] = 0.0
                
                regime_characteristics[regime_label] = characteristics
            else:
                regime_characteristics[regime_label] = {
                    'n_samples': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'center': None,
                    'stability': 0.0,
                    'volatility': 0.0
                }
        
        return regime_characteristics
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        if not self.clustering_results:
            return {"error": "No clustering performed yet"}
        
        return {
            "n_clusters": len(set(self.clustering_results.get('labels', []))),
            "method": self.clustering_results.get('method', 'unknown'),
            "metrics": self.clustering_results.get('metrics', {}),
            "regime_labels": self.clustering_results.get('regime_labels', []),
            "execution_time": self.clustering_results.get('execution_time', 0.0)
        }
    
    def update_clustering_results(self, results: Dict[str, Any]):
        """Update clustering results."""
        self.clustering_results = results