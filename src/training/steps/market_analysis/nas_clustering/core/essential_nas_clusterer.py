"""
Essential NAS Clusterer

Neural Architecture Search clustering for market regime detection.
"""

print("🔍 [ESSENTIAL_NAS_CLUSTERER] Loading Essential NAS Clusterer module")
print("🔍 [ESSENTIAL_NAS_CLUSTERER] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/essential_nas_clusterer.py")
print("🔍 [ESSENTIAL_NAS_CLUSTERER] Purpose: Neural Architecture Search clustering for market regime detection")
print("🔍 [ESSENTIAL_NAS_CLUSTERER] Status: Starting module import")

import numpy as np
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ NumPy imported successfully")

import pandas as pd
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Pandas imported successfully")

from typing import Dict, List, Any, Optional, Tuple, Union
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Typing imports completed")

import logging
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Logging imported successfully")

from sklearn.cluster import KMeans, AgglomerativeClustering
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Scikit-learn clustering algorithms imported")

from sklearn.mixture import GaussianMixture
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Gaussian Mixture Model imported")

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Clustering metrics imported")

from sklearn.preprocessing import StandardScaler
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ StandardScaler imported")

import torch
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ PyTorch imported successfully")

import torch.nn as nn
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ PyTorch neural network module imported")

import torch.optim as optim
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ PyTorch optimizer imported")

from collections import defaultdict
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Collections defaultdict imported")

import time
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Time module imported")

logger = logging.getLogger(__name__)
print("🔍 [ESSENTIAL_NAS_CLUSTERER] ✓ Logger initialized")
print("🔍 [ESSENTIAL_NAS_CLUSTERER] All imports completed successfully")

class EssentialNASClusterer:
    """Essential Neural Architecture Search Clusterer for market regime detection."""

    def __init__(self,
                 n_clusters: int = 5,
                 clustering_method: str = 'auto',
                 search_strategy: str = 'evolutionary',
                 max_iterations: int = 100,
                 device: str = 'cpu',
                 max_clusters_ratio: float = 0.1,
                 min_cluster_size_ratio: float = 0.01,
                 light_mode: bool = False,
                 max_cluster_size_ratio: float = 0.25,  # Reduced from 0.35 to prevent over-concentration
                 adaptive_clustering: bool = True,      # Enable adaptive clustering
                 concentration_threshold: float = 0.2): # More strict concentration threshold
        """Initialize Essential NAS Clusterer.

        Args:
            n_clusters: Number of clusters to find
            clustering_method: Clustering algorithm ('kmeans', 'gmm', 'dbscan', 'auto')
            search_strategy: NAS search strategy ('evolutionary', 'random', 'grid')
            max_iterations: Maximum search iterations
            device: Computation device
            max_clusters_ratio: Maximum clusters as ratio of data size (prevents over-clustering)
            min_cluster_size_ratio: Minimum cluster size as ratio of data size
            light_mode: If True, reduces minimum clusters to 3 and applies light mode constraints
            max_cluster_size_ratio: Maximum cluster size as ratio of data size (prevents over-concentration, default 25%)
            adaptive_clustering: Enable adaptive clustering to automatically adjust parameters
            concentration_threshold: Threshold for detecting over-concentration (default 20%)
        """
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Initializing EssentialNASClusterer")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Parameters received:")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - n_clusters: {n_clusters}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - clustering_method: {clustering_method}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - search_strategy: {search_strategy}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - max_iterations: {max_iterations}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - device: {device}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - max_clusters_ratio: {max_clusters_ratio}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - min_cluster_size_ratio: {min_cluster_size_ratio}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - light_mode: {light_mode}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - max_cluster_size_ratio: {max_cluster_size_ratio}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - adaptive_clustering: {adaptive_clustering}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT]   - concentration_threshold: {concentration_threshold}")
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Setting instance variables...")
        self.n_clusters = n_clusters
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ n_clusters set to: {self.n_clusters}")

        self.clustering_method = clustering_method
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ clustering_method set to: {self.clustering_method}")

        self.search_strategy = search_strategy
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ search_strategy set to: {self.search_strategy}")

        self.max_iterations = max_iterations
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ max_iterations set to: {self.max_iterations}")

        self.device = device
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ device set to: {self.device}")

        self.max_clusters_ratio = max_clusters_ratio
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ max_clusters_ratio set to: {self.max_clusters_ratio}")

        self.min_cluster_size_ratio = min_cluster_size_ratio
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ min_cluster_size_ratio set to: {self.min_cluster_size_ratio}")

        self.light_mode = light_mode
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ light_mode set to: {self.light_mode}")

        self.max_cluster_size_ratio = max_cluster_size_ratio
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ max_cluster_size_ratio set to: {self.max_cluster_size_ratio}")

        self.adaptive_clustering = adaptive_clustering
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ adaptive_clustering set to: {self.adaptive_clustering}")

        self.concentration_threshold = concentration_threshold
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ concentration_threshold set to: {self.concentration_threshold}")

        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Initializing clustering components...")
        # Clustering components
        self.scaler = StandardScaler()
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ StandardScaler initialized")

        self.clusterer = None
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ clusterer set to None")

        self.best_clusterer = None
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ best_clusterer set to None")

        self.best_score = -np.inf
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ best_score initialized to: {self.best_score}")

        self.best_params = None
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ best_params set to None")

        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Initializing search state...")
        # Search state
        self.search_history = []
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ search_history initialized as empty list")

        self.architecture_pool = []
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ architecture_pool initialized as empty list")

        self.performance_scores = []
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ performance_scores initialized as empty list")

        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Initializing results storage...")
        # Results
        self.cluster_labels = None
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ cluster_labels set to None")

        self.cluster_centers = None
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ cluster_centers set to None")

        self.cluster_metrics = {}
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ cluster_metrics initialized as empty dict")

        self.is_fitted = False
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] ✓ is_fitted set to False")

        print("🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Initialization complete!")
        logger.info(f"EssentialNASClusterer initialized with n_clusters={n_clusters}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_INIT] Logger info: EssentialNASClusterer initialized with n_clusters={n_clusters}")

    def search(self,
               data: np.ndarray,
               target: Optional[np.ndarray] = None,
               search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for optimal clustering architecture using NAS methodology.

        Args:
            data: Input data for clustering
            target: Optional target values for supervised clustering
            search_space: Optional custom search space

        Returns:
            Dictionary containing search results and best architecture
        """
        print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Starting NAS search for clustering")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data shape: {data.shape}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data type: {type(data)}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data dtype: {data.dtype}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data min: {np.min(data):.6f}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data max: {np.max(data):.6f}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data mean: {np.mean(data):.6f}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Data std: {np.std(data):.6f}")

        if target is not None:
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Target provided - shape: {target.shape}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Target type: {type(target)}")
        else:
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] No target provided - unsupervised clustering")

        if search_space is not None:
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Custom search space provided: {search_space}")
        else:
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Using default search space")

        logger.info(f"Starting NAS search for clustering with data shape: {data.shape}")
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Logger info: Starting NAS search for clustering with data shape: {data.shape}")

        start_time = time.time()
        print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Start time recorded: {start_time}")

        try:
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Starting try block")
            # Prepare data
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Fitting and transforming data with StandardScaler")
            data_scaled = self.scaler.fit_transform(data)
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Data scaled - shape: {data_scaled.shape}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Scaled data min: {np.min(data_scaled):.6f}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Scaled data max: {np.max(data_scaled):.6f}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Scaled data mean: {np.mean(data_scaled):.6f}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Scaled data std: {np.std(data_scaled):.6f}")

            # Define search space
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Defining search space...")
            if search_space is None:
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Using default search space")
                search_space = self._get_default_search_space()
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Default search space created: {search_space}")
            else:
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Using provided search space: {search_space}")

            # Apply adaptive clustering if enabled
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Adaptive clustering enabled: {self.adaptive_clustering}")
            if self.adaptive_clustering:
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Applying adaptive clustering preprocessing...")
                data_scaled, search_space = self._apply_adaptive_clustering(data_scaled, search_space)
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Adaptive clustering applied")
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Updated data shape: {data_scaled.shape}")
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Updated search space: {search_space}")
            else:
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Adaptive clustering disabled - using original data and search space")

            # Perform search based on strategy
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Selecting search strategy: {self.search_strategy}")
            if self.search_strategy == 'evolutionary':
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Using evolutionary search strategy")
                results = self._evolutionary_search(data_scaled, target, search_space)
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Evolutionary search completed")
            elif self.search_strategy == 'random':
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Using random search strategy")
                results = self._random_search(data_scaled, target, search_space)
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Random search completed")
            elif self.search_strategy == 'grid':
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Using grid search strategy")
                results = self._grid_search(data_scaled, target, search_space)
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Grid search completed")
            else:
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ❌ Unknown search strategy: {self.search_strategy}")
                raise ValueError(f"Unknown search strategy: {self.search_strategy}")

            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Search results: {results}")

            # Fit best clusterer with over-clustering validation
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Checking for best clusterer...")
            if self.best_clusterer is not None:
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Best clusterer found - fitting final model")
                self.clusterer = self.best_clusterer
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Clusterer assigned to self.clusterer")

                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Fitting and predicting cluster labels...")
                self.cluster_labels = self.clusterer.fit_predict(data_scaled)
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Cluster labels generated - shape: {self.cluster_labels.shape}")
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Unique labels: {np.unique(self.cluster_labels)}")
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Number of clusters found: {len(np.unique(self.cluster_labels))}")

                # Validate clustering results to prevent over-clustering
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Validating clustering results...")
                validated_labels = self._validate_clustering_results(data_scaled, self.cluster_labels)
                if validated_labels is not None:
                    print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Validation returned labels - updating cluster_labels")
                    self.cluster_labels = validated_labels
                    print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Updated unique labels: {np.unique(self.cluster_labels)}")
                else:
                    print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Validation returned None - keeping original labels")

                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Calculating cluster centers...")
                self.cluster_centers = self._get_cluster_centers(data_scaled, self.cluster_labels)
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Cluster centers calculated - shape: {self.cluster_centers.shape}")

                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Calculating cluster metrics...")
                self.cluster_metrics = self._calculate_cluster_metrics(data_scaled, self.cluster_labels)
                print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Cluster metrics calculated: {self.cluster_metrics}")

                self.is_fitted = True
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Model marked as fitted")
            else:
                print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ❌ No best clusterer found - search may have failed")

            search_time = time.time() - start_time
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Search completed in {search_time:.2f} seconds")

            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Updating results with final metrics...")
            n_clusters_found = len(np.unique(self.cluster_labels)) if self.cluster_labels is not None else 0
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Clusters found: {n_clusters_found}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Best score: {self.best_score}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Best params: {self.best_params}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Search history length: {len(self.search_history)}")

            results.update({
                'search_time': search_time,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'n_clusters_found': n_clusters_found,
                'cluster_metrics': self.cluster_metrics,
                'search_history': self.search_history
            })

            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Final results: {results}")
            logger.info(f"NAS search completed in {search_time:.2f}s with best score: {self.best_score:.4f}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Logger info: NAS search completed in {search_time:.2f}s with best score: {self.best_score:.4f}")
            print("🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ✓ Search method completed successfully")
            return results

        except Exception as e:
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] ❌ Exception occurred: {e}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Exception type: {type(e)}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Search time before error: {time.time() - start_time:.2f}s")
            logger.error(f"NAS search failed: {e}")
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Logger error: NAS search failed: {e}")

            error_result = {
                'success': False,
                'error': str(e),
                'search_time': time.time() - start_time
            }
            print(f"🔍 [ESSENTIAL_NAS_CLUSTERER_SEARCH] Returning error result: {error_result}")
            return error_result

    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space for clustering algorithms with over-clustering prevention."""
        # Calculate maximum clusters based on data size to prevent over-clustering
        # For early regime discovery, allow more clusters for better granularity
        max_clusters = max(2, min(15, int(self.n_clusters * 3)))

        # Adjust minimum clusters based on light mode
        min_clusters = 3 if self.light_mode else 2

        return {
            'clustering_method': ['kmeans', 'gmm', 'agglomerative'],
            'n_clusters': list(range(min_clusters, max_clusters + 1)),
            'kmeans_init': ['k-means++', 'random'],
            'kmeans_n_init': [10, 20],
            'gmm_covariance_type': ['full', 'tied', 'diag'],
            'agglomerative_linkage': ['ward', 'complete', 'average']
        }

    def _apply_adaptive_clustering(self, data: np.ndarray, search_space: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply adaptive clustering to improve cluster separability."""
        logger.info("Applying adaptive clustering preprocessing")

        try:
            n_samples = len(data)
            n_features = data.shape[1]

            # Calculate data characteristics
            data_variance = np.var(data, axis=0).mean()
            data_skewness = np.mean([abs(np.mean((data[:, i] - np.mean(data[:, i]))**3) / (np.std(data[:, i])**3)) for i in range(n_features)])
            data_kurtosis = np.mean([np.mean((data[:, i] - np.mean(data[:, i]))**4) / (np.std(data[:, i])**4) for i in range(n_features)])

            logger.info(f"Data characteristics - Variance: {data_variance:.4f}, Skewness: {data_skewness:.4f}, Kurtosis: {data_kurtosis:.4f}")

            # Apply feature scaling based on data characteristics
            if data_skewness > 1.0 or data_kurtosis > 3.0:
                # Use robust scaling for skewed data
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                data_scaled = scaler.fit_transform(data)
                logger.info("Applied RobustScaler for skewed data")
            elif data_variance > 10.0:
                # Use StandardScaler for high variance data
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                logger.info("Applied StandardScaler for high variance data")
            else:
                # Use MinMaxScaler for normalized data
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data)
                logger.info("Applied MinMaxScaler for normalized data")

            # Apply dimensionality reduction if needed
            if n_features > 50:
                from sklearn.decomposition import PCA
                n_components = min(20, n_features // 2)
                pca = PCA(n_components=n_components)
                data_scaled = pca.fit_transform(data_scaled)
                logger.info(f"Applied PCA: {n_features} -> {data_scaled.shape[1]} components")

            # Adjust search space based on data characteristics
            if data_skewness > 2.0:
                # Use more clusters for highly skewed data
                min_clusters = max(3, min_clusters)
                max_clusters = min(max_clusters, n_samples // 5)
                search_space['n_clusters'] = list(range(min_clusters, max_clusters + 1))
                logger.info(f"Adjusted cluster range for skewed data: {min_clusters}-{max_clusters}")

            if data_kurtosis > 5.0:
                # Use GMM for high kurtosis data (heavy tails)
                search_space['clustering_method'] = ['gmm', 'kmeans']
                logger.info("Prioritized GMM for high kurtosis data")

            return data_scaled, search_space

        except Exception as e:
            logger.warning(f"Adaptive clustering preprocessing failed: {e}")
            return data, search_space

    def _evolutionary_search(self,
                           data: np.ndarray,
                           target: Optional[np.ndarray],
                           search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform evolutionary search for optimal clustering architecture."""
        logger.info("Starting evolutionary search")

        # Initialize population
        population_size = 20
        population = self._generate_random_population(population_size, search_space)

        # Initialize generation counter
        generation = 0

        for generation in range(self.max_iterations // population_size):
            # Evaluate population
            scores = []
            for individual in population:
                score = self._evaluate_architecture(data, target, individual)
                scores.append(score)
                self.search_history.append({
                    'generation': generation,
                    'individual': individual,
                    'score': score
                })

            # Update best
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx]
                self.best_params = population[best_idx]
                self.best_clusterer = self._create_clusterer(self.best_params)

            # Selection and reproduction
            population = self._evolve_population(population, scores)

            logger.info(f"Generation {generation}: Best score = {self.best_score:.4f}")

        return {
            'success': True,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'generations': generation + 1
        }

    def _random_search(self,
                      data: np.ndarray,
                      target: Optional[np.ndarray],
                      search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform random search for optimal clustering architecture."""
        logger.info("Starting random search")

        for iteration in range(self.max_iterations):
            # Generate random architecture
            params = self._sample_random_params(search_space)

            # Evaluate architecture
            score = self._evaluate_architecture(data, target, params)

            self.search_history.append({
                'iteration': iteration,
                'params': params,
                'score': score
            })

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_clusterer = self._create_clusterer(self.best_params)

            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best score = {self.best_score:.4f}")

        return {
            'success': True,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'iterations': self.max_iterations
        }

    def _grid_search(self,
                    data: np.ndarray,
                    target: Optional[np.ndarray],
                    search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform grid search for optimal clustering architecture."""
        logger.info("Starting grid search")

        # Generate grid of parameters
        param_grid = self._generate_param_grid(search_space)

        for i, params in enumerate(param_grid):
            # Evaluate architecture
            score = self._evaluate_architecture(data, target, params)

            self.search_history.append({
                'iteration': i,
                'params': params,
                'score': score
            })

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_clusterer = self._create_clusterer(self.best_params)

            if i % 10 == 0:
                logger.info(f"Grid point {i}: Best score = {self.best_score:.4f}")

        return {
            'success': True,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'grid_points': len(param_grid)
        }

    def _generate_random_population(self, size: int, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate random population for evolutionary search."""
        population = []
        for _ in range(size):
            individual = self._sample_random_params(search_space)
            population.append(individual)
        return population

    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # Range parameter
                if isinstance(values[0], int):
                    params[key] = np.random.randint(values[0], values[1] + 1)
                else:
                    params[key] = np.random.uniform(values[0], values[1])
        return params

    def _generate_param_grid(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        from itertools import product

        # Convert search space to lists
        param_lists = []
        param_names = []

        for key, values in search_space.items():
            if isinstance(values, list):
                param_lists.append(values)
                param_names.append(key)

        # Generate all combinations
        param_combinations = list(product(*param_lists))

        # Convert to list of dictionaries
        param_grid = []
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            param_grid.append(params)

        return param_grid

    def _evaluate_architecture(self,
                              data: np.ndarray,
                              target: Optional[np.ndarray],
                              params: Dict[str, Any]) -> float:
        """Evaluate clustering architecture with over-clustering prevention."""
        try:
            # Create clusterer with given parameters
            clusterer = self._create_clusterer(params)

            # Fit and predict
            labels = clusterer.fit_predict(data)

            # Handle case where all points are in one cluster
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                return -np.inf

            # Check cluster size distribution and warn about issues
            n_samples = len(data)
            min_cluster_size = int(n_samples * self.min_cluster_size_ratio)
            max_clusters_allowed = int(n_samples * self.max_clusters_ratio)
            max_cluster_size = int(n_samples * self.max_cluster_size_ratio)

            # Check for over-clustering
            cluster_sizes = [np.sum(labels == label) for label in np.unique(labels)]
            small_clusters = sum(1 for size in cluster_sizes if size < min_cluster_size)

            if n_clusters > max_clusters_allowed:
                logger.warning(f"⚠️ Over-clustering detected: {n_clusters} clusters > {max_clusters_allowed} allowed")

            if small_clusters > 0:
                logger.warning(f"⚠️ Small clusters detected: {small_clusters} clusters < {min_cluster_size} samples")

            # Check for over-concentration (no cluster should have more than threshold of data)
            large_clusters = sum(1 for size in cluster_sizes if size > max_cluster_size)
            concentration_penalty = 0.0

            if large_clusters > 0:
                # Calculate concentration penalty - more severe for larger over-concentration
                max_concentration = max(size / n_samples for size in cluster_sizes)
                concentration_penalty = (max_concentration - self.concentration_threshold) * 10.0
                logger.warning(f"⚠️ Over-concentration detected: {large_clusters} clusters > {max_cluster_size} samples (>{self.concentration_threshold*100:.1f}% of data)")
                logger.warning(f"   -> Concentration penalty: {concentration_penalty:.4f}")

            # Calculate clustering metrics
            silhouette = silhouette_score(data, labels)
            calinski_harabasz = calinski_harabasz_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)

            # Calculate cluster balance metric (lower variance in cluster sizes is better)
            cluster_size_variance = np.var(cluster_sizes)
            balance_score = 1.0 / (1.0 + cluster_size_variance / (n_samples / n_clusters))

            # Combined score with concentration penalty and balance reward
            score = silhouette + (calinski_harabasz / 1000) - davies_bouldin + balance_score - concentration_penalty

            return score

        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return -np.inf

    def _create_clusterer(self, params: Dict[str, Any]):
        """Create clustering algorithm with given parameters."""
        method = params.get('clustering_method', 'kmeans')

        if method == 'kmeans':
            return KMeans(
                n_clusters=params.get('n_clusters', self.n_clusters),
                init=params.get('kmeans_init', 'k-means++'),
                n_init=params.get('kmeans_n_init', 10),
                random_state=42
            )
        elif method == 'gmm':
            return GaussianMixture(
                n_components=params.get('n_clusters', self.n_clusters),
                covariance_type=params.get('gmm_covariance_type', 'full'),
                random_state=42
            )
        elif method == 'agglomerative':
            return AgglomerativeClustering(
                n_clusters=params.get('n_clusters', self.n_clusters),
                linkage=params.get('agglomerative_linkage', 'ward')
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _evolve_population(self, population: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """Evolve population using genetic operators."""
        # Selection: Keep top 50% and replace bottom 50%
        sorted_indices = np.argsort(scores)[::-1]
        elite_size = len(population) // 2

        new_population = []

        # Keep elite
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])

        # Generate offspring
        for _ in range(len(population) - elite_size):
            # Tournament selection
            parent1 = self._tournament_selection(population, scores)
            parent2 = self._tournament_selection(population, scores)

            # Crossover and mutation
            offspring = self._crossover_and_mutate(parent1, parent2)
            new_population.append(offspring)

        return new_population

    def _tournament_selection(self, population: List[Dict[str, Any]], scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]

    def _crossover_and_mutate(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover and mutation operations."""
        offspring = parent1.copy()

        # Crossover: randomly inherit from parent2
        for key in offspring:
            if np.random.random() < 0.5:
                offspring[key] = parent2[key]

        # Mutation: randomly change some parameters
        for key in offspring:
            if np.random.random() < 0.1:  # 10% mutation rate
                # Simple mutation - could be enhanced
                if isinstance(offspring[key], str):
                    # For string parameters, randomly select from common values
                    if key == 'clustering_method':
                        offspring[key] = np.random.choice(['kmeans', 'gmm', 'dbscan', 'agglomerative'])
                elif isinstance(offspring[key], int):
                    # For integer parameters, add small random change
                    offspring[key] = max(1, offspring[key] + np.random.randint(-2, 3))

        return offspring

    def _get_cluster_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        unique_labels = np.unique(labels)
        centers = []

        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            cluster_data = data[labels == label]
            center = np.mean(cluster_data, axis=0)
            centers.append(center)

        return np.array(centers) if centers else np.array([])

    def _validate_clustering_results(self, data: np.ndarray, labels: np.ndarray) -> Optional[np.ndarray]:
        """Validate clustering results and warn about issues without automatic correction."""
        try:
            n_samples = len(data)
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            # Check for over-clustering
            max_clusters_allowed = int(n_samples * self.max_clusters_ratio)
            min_cluster_size = int(n_samples * self.min_cluster_size_ratio)
            max_cluster_size = int(n_samples * self.max_cluster_size_ratio)

            # Adjust minimum clusters for light mode
            min_clusters_required = 3 if self.light_mode else 2

            # Check for insufficient clusters
            if n_clusters < min_clusters_required:
                logger.warning(f"⚠️ Insufficient clusters: {n_clusters} < {min_clusters_required} required")
                if self.light_mode:
                    logger.warning("   -> Consider increasing n_clusters or disabling light mode")
                else:
                    logger.warning("   -> Consider increasing n_clusters parameter")

            # Check for over-clustering
            if n_clusters > max_clusters_allowed:
                logger.warning(f"⚠️ Over-clustering detected: {n_clusters} clusters > {max_clusters_allowed} allowed")
                logger.warning("   -> Consider reducing n_clusters or increasing max_clusters_ratio")

            # Check for small clusters
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            small_clusters = [label for label, size in zip(unique_labels, cluster_sizes)
                            if size < min_cluster_size]

            if small_clusters:
                logger.warning(f"⚠️ Small clusters detected: {len(small_clusters)} clusters < {min_cluster_size} samples")
                logger.warning("   -> Consider reducing n_clusters or adjusting min_cluster_size_ratio")

            # Check for over-concentration (no cluster should have more than 35% of data)
            large_clusters = [label for label, size in zip(unique_labels, cluster_sizes)
                            if size > max_cluster_size]

            if large_clusters:
                logger.warning(f"⚠️ Over-concentration detected: {len(large_clusters)} clusters > {max_cluster_size} samples (>{self.max_cluster_size_ratio*100:.1f}% of data)")
                logger.warning("   -> Consider increasing n_clusters or adjusting max_cluster_size_ratio")

            # Check for too many small clusters
            small_clusters_count = sum(1 for size in cluster_sizes if size < min_cluster_size)

            if small_clusters_count > n_clusters * 0.5:  # More than 50% are small
                logger.warning(f"⚠️ Too many small clusters: {small_clusters_count}/{n_clusters} ({small_clusters_count/n_clusters*100:.1f}%)")
                logger.warning("   -> Consider reducing n_clusters or adjusting clustering parameters")

            # Always return original labels - no automatic correction
            return labels

        except Exception as e:
            logger.error(f"Clustering validation failed: {e}")
            return labels  # Return original labels if validation fails

    def _calculate_cluster_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive clustering metrics."""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            if n_clusters < 2:
                return {'error': 'Insufficient clusters for metrics'}

            metrics = {
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_score(data, labels),
                'calinski_harabasz_score': calinski_harabasz_score(data, labels),
                'davies_bouldin_score': davies_bouldin_score(data, labels)
            }

            # Additional metrics
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            metrics.update({
                'avg_cluster_size': np.mean(cluster_sizes),
                'cluster_size_std': np.std(cluster_sizes),
                'min_cluster_size': np.min(cluster_sizes),
                'max_cluster_size': np.max(cluster_sizes)
            })

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate cluster metrics: {e}")
            return {'error': str(e)}

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Clusterer must be fitted before prediction")

        data_scaled = self.scaler.transform(data)
        return self.clusterer.predict(data_scaled)

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the fitted clusters."""
        if not self.is_fitted:
            return {'error': 'Clusterer not fitted'}

        return {
            'n_clusters': len(np.unique(self.cluster_labels)),
            'cluster_centers': self.cluster_centers.tolist() if self.cluster_centers is not None else [],
            'cluster_metrics': self.cluster_metrics,
            'best_params': self.best_params,
            'best_score': self.best_score
        }

    def save_model(self, filepath: str):
        """Save the clusterer model."""
        try:
            import joblib

            model_data = {
                'scaler': self.scaler,
                'clusterer': self.clusterer,
                'best_clusterer': self.best_clusterer,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'cluster_labels': self.cluster_labels,
                'cluster_centers': self.cluster_centers,
                'cluster_metrics': self.cluster_metrics,
                'search_history': self.search_history,
                'is_fitted': self.is_fitted,
                'n_clusters': self.n_clusters,
                'clustering_method': self.clustering_method,
                'search_strategy': self.search_strategy,
                'max_iterations': self.max_iterations,
                'device': self.device,
                'light_mode': self.light_mode,
                'max_cluster_size_ratio': self.max_cluster_size_ratio,
                'adaptive_clustering': self.adaptive_clustering,
                'concentration_threshold': self.concentration_threshold
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Clusterer saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save clusterer: {e}")

    def load_model(self, filepath: str):
        """Load the clusterer model."""
        try:
            import joblib

            model_data = joblib.load(filepath)

            self.scaler = model_data['scaler']
            self.clusterer = model_data['clusterer']
            self.best_clusterer = model_data['best_clusterer']
            self.best_score = model_data['best_score']
            self.best_params = model_data['best_params']
            self.cluster_labels = model_data['cluster_labels']
            self.cluster_centers = model_data['cluster_centers']
            self.cluster_metrics = model_data['cluster_metrics']
            self.search_history = model_data['search_history']
            self.is_fitted = model_data['is_fitted']
            self.n_clusters = model_data['n_clusters']
            self.clustering_method = model_data['clustering_method']
            self.search_strategy = model_data['search_strategy']
            self.max_iterations = model_data['max_iterations']
            self.device = model_data['device']
            self.light_mode = model_data.get('light_mode', False)
            self.max_cluster_size_ratio = model_data.get('max_cluster_size_ratio', 0.25)
            self.adaptive_clustering = model_data.get('adaptive_clustering', True)
            self.concentration_threshold = model_data.get('concentration_threshold', 0.2)

            logger.info(f"Clusterer loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load clusterer: {e}")
            raise
