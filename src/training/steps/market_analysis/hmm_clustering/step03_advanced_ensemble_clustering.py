#!/usr/bin/env python3
"""Advanced Ensemble Clustering with Hierarchical Consensus and Dynamic Weighting.

This module provides a superior alternative to the basic ensemble clustering approach
using hierarchical consensus, dynamic weighting, and advanced clustering algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, Birch, OPTICS
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import centralized systems
from .step03_imports import get_import_manager, safe_import, check_feature_availability
from .step03_config import Step03Config
from .step03_memory_manager import get_memory_manager, memory_aware_processing

# Import optional dependencies
hmmlearn = safe_import('hmmlearn')
sklearn = safe_import('sklearn')


@dataclass
class ClusteringAlgorithm:
    """Configuration for a clustering algorithm."""
    name: str
    algorithm_class: Any
    parameter_ranges: Dict[str, Any]
    weight: float = 1.0
    stability_threshold: float = 0.7


class AdvancedEnsembleClustering:
    """Advanced ensemble clustering with hierarchical consensus and dynamic weighting."""
    
    def __init__(self, config: Step03Config):
        self.config = config
        self.logger = logging.getLogger('AdvancedEnsembleClustering')
        self.memory_manager = get_memory_manager(config.memory.__dict__)
        
        # Initialize clustering algorithms
        self._initialize_clustering_algorithms()
        
        # Results storage
        self.clustering_results = {}
        self.consensus_matrix = None
        self.final_regimes = None
        self.algorithm_weights = {}
        
    def _initialize_clustering_algorithms(self):
        """Initialize comprehensive set of clustering algorithms."""
        self.algorithms = {}
        
        # HMM Clustering
        if hmmlearn:
            self.algorithms['hmm'] = ClusteringAlgorithm(
                name='hmm',
                algorithm_class=hmmlearn.hmm.GaussianHMM,
                parameter_ranges={
                    'n_components': (2, 8),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'n_iter': (50, 200),
                    'tol': (1e-6, 1e-2),
                    'reg_covar': (1e-7, 1e-2)
                },
                weight=0.3,
                stability_threshold=0.8
            )
        
        # K-Means variants
        if sklearn:
            from sklearn.cluster import KMeans, MiniBatchKMeans
            
            self.algorithms['kmeans'] = ClusteringAlgorithm(
                name='kmeans',
                algorithm_class=KMeans,
                parameter_ranges={
                    'n_clusters': (5, 30),
                    'n_init': (10, 20),
                    'max_iter': (100, 500),
                    'algorithm': ['lloyd', 'elkan']
                },
                weight=0.2,
                stability_threshold=0.7
            )
            
            self.algorithms['mini_batch_kmeans'] = ClusteringAlgorithm(
                name='mini_batch_kmeans',
                algorithm_class=MiniBatchKMeans,
                parameter_ranges={
                    'n_clusters': (5, 30),
                    'batch_size': (100, 1000),
                    'max_iter': (100, 500)
                },
                weight=0.15,
                stability_threshold=0.6
            )
        
        # Gaussian Mixture Models
        if sklearn:
            self.algorithms['gaussian_mixture'] = ClusteringAlgorithm(
                name='gaussian_mixture',
                algorithm_class=GaussianMixture,
                parameter_ranges={
                    'n_components': (2, 15),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'init_params': ['kmeans', 'random'],
                    'max_iter': (50, 200)
                },
                weight=0.2,
                stability_threshold=0.75
            )
            
            self.algorithms['bayesian_gaussian_mixture'] = ClusteringAlgorithm(
                name='bayesian_gaussian_mixture',
                algorithm_class=BayesianGaussianMixture,
                parameter_ranges={
                    'n_components': (2, 15),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'init_params': ['kmeans', 'random'],
                    'max_iter': (50, 200)
                },
                weight=0.25,
                stability_threshold=0.8
            )
        
        # Density-based clustering
        if sklearn:
            from sklearn.cluster import DBSCAN
            
            self.algorithms['dbscan'] = ClusteringAlgorithm(
                name='dbscan',
                algorithm_class=DBSCAN,
                parameter_ranges={
                    'eps': (0.1, 2.0),
                    'min_samples': (5, 50),
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight=0.15,
                stability_threshold=0.6
            )
            
            self.algorithms['optics'] = ClusteringAlgorithm(
                name='optics',
                algorithm_class=OPTICS,
                parameter_ranges={
                    'min_samples': (5, 50),
                    'max_eps': (0.5, 5.0),
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight=0.1,
                stability_threshold=0.5
            )
        
        # Hierarchical clustering
        if sklearn:
            self.algorithms['agglomerative'] = ClusteringAlgorithm(
                name='agglomerative',
                algorithm_class=AgglomerativeClustering,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight=0.2,
                stability_threshold=0.7
            )
            
            self.algorithms['birch'] = ClusteringAlgorithm(
                name='birch',
                algorithm_class=Birch,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'threshold': (0.1, 2.0),
                    'branching_factor': (10, 100)
                },
                weight=0.15,
                stability_threshold=0.6
            )
        
        # Spectral clustering
        if sklearn:
            self.algorithms['spectral'] = ClusteringAlgorithm(
                name='spectral',
                algorithm_class=SpectralClustering,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'affinity': ['rbf', 'nearest_neighbors', 'polynomial'],
                    'gamma': (0.1, 10.0),
                    'n_neighbors': (5, 50)
                },
                weight=0.2,
                stability_threshold=0.7
            )
    
    def _generate_parameter_combinations(self, algorithm: ClusteringAlgorithm, n_combinations: int = 10) -> List[Dict]:
        """Generate parameter combinations for an algorithm."""
        combinations = []
        
        for _ in range(n_combinations):
            params = {}
            for param_name, param_range in algorithm.parameter_ranges.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
            
            # Add common parameters
            params['random_state'] = 42
            params['n_jobs'] = -1
            
            combinations.append(params)
        
        return combinations
    
    def _evaluate_clustering_stability(self, features: np.ndarray, algorithm: ClusteringAlgorithm, 
                                     params: Dict, n_bootstrap: int = 5) -> float:
        """Evaluate clustering stability using bootstrap sampling."""
        try:
            stability_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                n_samples = len(features)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_features = features[bootstrap_indices]
                
                # Train clustering model
                model = algorithm.algorithm_class(**params)
                
                if hasattr(model, 'fit_predict'):
                    labels = model.fit_predict(bootstrap_features)
                else:
                    model.fit(bootstrap_features)
                    labels = model.predict(bootstrap_features)
                
                # Calculate stability metric (adjusted rand index with original)
                if len(np.unique(labels)) > 1:
                    # Use silhouette score as stability metric
                    from sklearn.metrics import silhouette_score
                    try:
                        score = silhouette_score(bootstrap_features, labels)
                        stability_scores.append(score)
                    except:
                        stability_scores.append(0.0)
                else:
                    stability_scores.append(0.0)
            
            return np.mean(stability_scores)
            
        except Exception as e:
            self.logger.warning(f"Stability evaluation failed for {algorithm.name}: {e}")
            return 0.0
    
    def _calculate_algorithm_weights(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic weights for each algorithm based on performance."""
        weights = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                # Generate parameter combinations
                param_combinations = self._generate_parameter_combinations(algorithm, n_combinations=5)
                
                # Evaluate stability for each combination
                stability_scores = []
                for params in param_combinations:
                    stability = self._evaluate_clustering_stability(features, algorithm, params)
                    stability_scores.append(stability)
                
                # Calculate average stability
                avg_stability = np.mean(stability_scores)
                
                # Weight based on stability and base weight
                weight = algorithm.weight * avg_stability
                weights[name] = weight
                
                self.logger.debug(f"Algorithm {name}: stability={avg_stability:.3f}, weight={weight:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {name}: {e}")
                weights[name] = algorithm.weight * 0.1  # Minimal weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _build_consensus_matrix(self, features: np.ndarray) -> np.ndarray:
        """Build consensus matrix from multiple clustering results."""
        n_samples = len(features)
        consensus_matrix = np.zeros((n_samples, n_samples))
        total_weight = 0
        
        for name, algorithm in self.algorithms.items():
            if name not in self.algorithm_weights:
                continue
            
            weight = self.algorithm_weights[name]
            if weight < 0.01:  # Skip very low weight algorithms
                continue
            
            try:
                # Generate multiple parameter combinations
                param_combinations = self._generate_parameter_combinations(algorithm, n_combinations=3)
                
                for params in param_combinations:
                    try:
                        # Train model
                        model = algorithm.algorithm_class(**params)
                        
                        if hasattr(model, 'fit_predict'):
                            labels = model.fit_predict(features)
                        else:
                            model.fit(features)
                            labels = model.predict(features)
                        
                        # Build co-occurrence matrix
                        for i in range(n_samples):
                            for j in range(n_samples):
                                if labels[i] == labels[j]:
                                    consensus_matrix[i, j] += weight
                        
                        total_weight += weight
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to run {name} with params {params}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.warning(f"Failed to process algorithm {name}: {e}")
                continue
        
        # Normalize consensus matrix
        if total_weight > 0:
            consensus_matrix /= total_weight
        
        return consensus_matrix
    
    def _hierarchical_consensus_clustering(self, consensus_matrix: np.ndarray, 
                                         n_clusters_range: Tuple[int, int] = (2, 10)) -> np.ndarray:
        """Perform hierarchical consensus clustering."""
        try:
            # Convert consensus matrix to distance matrix
            distance_matrix = 1 - consensus_matrix
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # Find optimal number of clusters
            best_score = -np.inf
            best_labels = None
            best_n_clusters = 2
            
            for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
                
                # Calculate clustering quality
                if len(np.unique(labels)) > 1:
                    from sklearn.metrics import silhouette_score
                    try:
                        # Use a subset for silhouette score calculation
                        n_samples = len(consensus_matrix)
                        subset_size = min(1000, n_samples)
                        subset_indices = np.random.choice(n_samples, subset_size, replace=False)
                        
                        subset_consensus = consensus_matrix[np.ix_(subset_indices, subset_indices)]
                        subset_labels = labels[subset_indices]
                        
                        # Convert to distance matrix for silhouette score
                        subset_distance = 1 - subset_consensus
                        
                        # Calculate silhouette score
                        score = silhouette_score(subset_distance, subset_labels, metric='precomputed')
                        
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_n_clusters = n_clusters
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate silhouette score for {n_clusters} clusters: {e}")
                        continue
            
            self.logger.info(f"Best hierarchical clustering: {best_n_clusters} clusters, score: {best_score:.3f}")
            
            return best_labels if best_labels is not None else np.zeros(len(consensus_matrix), dtype=int)
            
        except Exception as e:
            self.logger.error(f"Hierarchical consensus clustering failed: {e}")
            # Fallback to simple consensus voting
            return self._simple_consensus_voting(consensus_matrix)
    
    def _simple_consensus_voting(self, consensus_matrix: np.ndarray) -> np.ndarray:
        """Simple consensus voting as fallback."""
        try:
            # Use spectral clustering on consensus matrix
            from sklearn.cluster import SpectralClustering
            
            # Estimate number of clusters
            n_clusters = min(8, max(2, len(consensus_matrix) // 100))
            
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            labels = spectral.fit_predict(consensus_matrix)
            return labels
            
        except Exception as e:
            self.logger.error(f"Simple consensus voting failed: {e}")
            return np.zeros(len(consensus_matrix), dtype=int)
    
    def _validate_regime_quality(self, features: np.ndarray, regimes: np.ndarray) -> Dict[str, float]:
        """Validate the quality of discovered regimes."""
        try:
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            if n_regimes < 2:
                return {'quality_score': 0.0, 'n_regimes': n_regimes}
            
            # Calculate various quality metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Silhouette score
            try:
                silhouette = silhouette_score(features, regimes)
            except:
                silhouette = 0.0
            
            # Calinski-Harabasz score
            try:
                calinski_harabasz = calinski_harabasz_score(features, regimes)
            except:
                calinski_harabasz = 0.0
            
            # Davies-Bouldin score
            try:
                davies_bouldin = davies_bouldin_score(features, regimes)
            except:
                davies_bouldin = float('inf')
            
            # Regime balance
            regime_sizes = [np.sum(regimes == regime) for regime in unique_regimes]
            regime_balance = 1 / (1 + np.std(regime_sizes) / np.mean(regime_sizes))
            
            # Regime stability (inverse of transition frequency)
            regime_changes = np.sum(np.diff(regimes) != 0)
            regime_stability = 1 / (1 + regime_changes / len(regimes))
            
            # Combined quality score
            quality_score = (
                0.3 * max(0, silhouette) +
                0.2 * min(1, calinski_harabasz / 1000) +
                0.2 * (1 / (1 + davies_bouldin)) +
                0.15 * regime_balance +
                0.15 * regime_stability
            )
            
            return {
                'quality_score': quality_score,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'regime_balance': regime_balance,
                'regime_stability': regime_stability,
                'n_regimes': n_regimes
            }
            
        except Exception as e:
            self.logger.error(f"Regime quality validation failed: {e}")
            return {'quality_score': 0.0, 'n_regimes': len(np.unique(regimes))}
    
    def ensemble_regime_detection(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform advanced ensemble regime detection."""
        self.logger.info("🚀 Starting advanced ensemble regime detection...")
        
        with memory_aware_processing("ensemble_clustering", self.config.memory.__dict__):
            # Step 1: Calculate dynamic algorithm weights
            self.logger.info("📊 Calculating dynamic algorithm weights...")
            self.algorithm_weights = self._calculate_algorithm_weights(features)
            
            # Step 2: Build consensus matrix
            self.logger.info("🔗 Building consensus matrix...")
            self.consensus_matrix = self._build_consensus_matrix(features)
            
            # Step 3: Hierarchical consensus clustering
            self.logger.info("🌳 Performing hierarchical consensus clustering...")
            self.final_regimes = self._hierarchical_consensus_clustering(self.consensus_matrix)
            
            # Step 4: Validate regime quality
            self.logger.info("✅ Validating regime quality...")
            quality_metrics = self._validate_regime_quality(features, self.final_regimes)
            
            # Compile results
            results = {
                'regimes': self.final_regimes,
                'consensus_matrix': self.consensus_matrix,
                'algorithm_weights': self.algorithm_weights,
                'quality_metrics': quality_metrics,
                'n_regimes': len(np.unique(self.final_regimes)),
                'regime_distribution': {
                    f'regime_{regime}': int(np.sum(self.final_regimes == regime))
                    for regime in np.unique(self.final_regimes)
                }
            }
            
            self.logger.info(f"✅ Advanced ensemble clustering completed")
            self.logger.info(f"   Discovered {results['n_regimes']} regimes")
            self.logger.info(f"   Quality score: {quality_metrics['quality_score']:.3f}")
            self.logger.info(f"   Silhouette score: {quality_metrics.get('silhouette_score', 0):.3f}")
            
            return self.final_regimes, results