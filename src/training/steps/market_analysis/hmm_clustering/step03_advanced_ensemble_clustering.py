from ..standardized_parquet_handler import standardized_parquet_handler
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

import warnings

import threading
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Performance optimization imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

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

@dataclass
class ParallelClusteringConfig:
    """Configuration for parallel clustering."""
    max_workers: int = None  # Auto-detect if None
    enable_early_consensus: bool = True
    consensus_threshold: float = 0.8
    min_algorithms_for_consensus: int = 3
    parallel_chunk_size: int = 1000
    thread_timeout_seconds: int = 300

@dataclass
class EarlyConsensusDetector:
    """Detector for early consensus in ensemble clustering."""

    def __init__(self, consensus_threshold: float = 0.8, min_algorithms: int = 3):
        self.consensus_threshold = consensus_threshold
        self.min_algorithms = min_algorithms
        self.algorithm_results = {}
        self.consensus_matrix = None

    def add_algorithm_result(self, algorithm_name: str, labels: np.ndarray) -> bool:
        """Add result from an algorithm and check for early consensus."""
        self.algorithm_results[algorithm_name] = labels

        if len(self.algorithm_results) < self.min_algorithms:
            return False

        # Build consensus matrix with current results
        self._build_partial_consensus_matrix()

        # Check for consensus
        consensus_score = self._calculate_consensus_score()
        return consensus_score >= self.consensus_threshold

    def _build_partial_consensus_matrix(self):
        """Build consensus matrix from current algorithm results."""
        n_samples = len(next(iter(self.algorithm_results.values())))
        n_algorithms = len(self.algorithm_results)

        self.consensus_matrix = np.zeros((n_samples, n_samples))

        algorithm_labels = list(self.algorithm_results.values())

        # Calculate co-occurrence matrix
        for i in range(n_samples):
            for j in range(n_samples):
                agreement_count = sum(1 for labels in algorithm_labels
                                    if labels[i] == labels[j])
                self.consensus_matrix[i, j] = agreement_count / n_algorithms

    def _calculate_consensus_score(self) -> float:
        """Calculate consensus score from current results."""
        if self.consensus_matrix is None:
            return 0.0

        # Calculate average consensus
        upper_triangle = np.triu(self.consensus_matrix, k=1)
        non_zero_elements = upper_triangle[upper_triangle > 0]
        return np.mean(non_zero_elements) if len(non_zero_elements) > 0 else 0.0

    def get_consensus_labels(self, n_clusters: int) -> np.ndarray:
        """Get consensus clustering labels."""
        if self.consensus_matrix is None:
            return np.zeros(len(next(iter(self.algorithm_results.values()))))

        # Use hierarchical clustering on consensus matrix
        from scipy.cluster.hierarchy import linkage, fcluster
        linkage_matrix = linkage(1 - self.consensus_matrix, method='average')
        return fcluster(linkage_matrix, n_clusters, criterion='maxclust')

class ParallelClusteringProcessor:
    """Parallel processor for ensemble clustering algorithms."""

    def __init__(self, config: ParallelClusteringConfig = None):
        self.config = config or ParallelClusteringConfig()
        self.max_workers = self.config.max_workers or min(8, (joblib.cpu_count() if JOBLIB_AVAILABLE else 4))
        self.logger = logging.getLogger('ParallelClusteringProcessor')

        # Thread safety
        self._lock = threading.Lock()
        self.completed_algorithms = 0
        self.early_consensus_detected = False

    def process_algorithms_parallel(self, algorithms: Dict[str, ClusteringAlgorithm],
                                  features: np.ndarray, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Process clustering algorithms in parallel with early consensus detection."""

        self.logger.info(f"🚀 Starting parallel clustering with {self.max_workers} workers")

        # Initialize early consensus detector
        consensus_detector = EarlyConsensusDetector(
            self.config.consensus_threshold,
            self.config.min_algorithms_for_consensus
        )

        results = {}
        algorithm_queue = list(algorithms.items())

        # Process algorithms in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            # Submit initial batch
            for algorithm_name, algorithm in algorithm_queue[:self.max_workers]:
                future = executor.submit(
                    self._process_single_algorithm,
                    algorithm_name, algorithm, features, data
                )
                futures[future] = (algorithm_name, algorithm)

            # Process results as they complete
            while futures:
                # Wait for any future to complete
                completed_futures, _ = concurrent.futures.wait(
                    futures.keys(),
                    timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                for future in completed_futures:
                    algorithm_name, algorithm = futures[future]

                    try:
                        result = future.result(timeout=self.config.thread_timeout_seconds)
                        results[algorithm_name] = result

                        # Check for early consensus
                        with self._lock:
                            self.completed_algorithms += 1
                            consensus_reached = consensus_detector.add_algorithm_result(
                                algorithm_name, result['labels']
                            )

                            if consensus_reached and not self.early_consensus_detected:
                                self.early_consensus_detected = True
                                self.logger.info("⏹️ Early consensus detected, stopping remaining algorithms")

                    except Exception as e:
                        self.logger.error(f"Algorithm {algorithm_name} failed: {e}")
                        results[algorithm_name] = {'error': str(e)}

                    del futures[future]

                    # Submit next algorithm if available and no early consensus
                    if (algorithm_queue and len(futures) < self.max_workers
                        and not self.early_consensus_detected):
                        next_name, next_algorithm = algorithm_queue.pop(0)
                        future = executor.submit(
                            self._process_single_algorithm,
                            next_name, next_algorithm, features, data
                        )
                        futures[future] = (next_name, next_algorithm)

                    # Stop if early consensus detected
                    if self.early_consensus_detected:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

        # Process final results
        final_results = self._process_parallel_results(results, consensus_detector)

        self.logger.info(f"✅ Parallel clustering completed: {len(results)} algorithms processed")
        if self.early_consensus_detected:
            self.logger.info("🎯 Early consensus optimization saved computational resources")

        return final_results

    def _process_single_algorithm(self, algorithm_name: str, algorithm: ClusteringAlgorithm,
                                features: np.ndarray, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Process a single clustering algorithm."""

        try:
            self.logger.debug(f"Processing algorithm: {algorithm_name}")

            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(algorithm)

            best_result = None
            best_score = -float('inf')

            # Try different parameter combinations
            for params in param_combinations:
                try:
                    # Apply clustering
                    labels = self._apply_clustering_algorithm(algorithm, features, params)

                    # Evaluate clustering quality
                    quality_score = self._evaluate_clustering_quality(labels, features)

                    if quality_score > best_score:
                        best_score = quality_score
                        best_result = {
                            'labels': labels,
                            'parameters': params,
                            'quality_score': quality_score,
                            'algorithm': algorithm_name
                        }

                except Exception as e:
                    self.logger.warning(f"Parameter combination failed for {algorithm_name}: {e}")
                    continue

            if best_result is None:
                raise RuntimeError(f"No valid results for algorithm {algorithm_name}")

            return best_result

        except Exception as e:
            self.logger.error(f"Algorithm {algorithm_name} processing failed: {e}")
            raise

    def _generate_parameter_combinations(self, algorithm: ClusteringAlgorithm,
                                       n_combinations: int = 5) -> List[Dict[str, Any]]:
        """Generate parameter combinations for an algorithm."""
        combinations = []

        # Simple parameter sampling - could be enhanced with more sophisticated sampling
        for _ in range(n_combinations):
            params = {}
            for param_name, param_range in algorithm.parameter_ranges.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Numeric range
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical
                    params[param_name] = np.random.choice(param_range)
                else:
                    params[param_name] = param_range

            combinations.append(params)

        return combinations

    def _apply_clustering_algorithm(self, algorithm: ClusteringAlgorithm,
                                  features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply clustering algorithm with given parameters."""

        if algorithm.name == 'hmm' and hmmlearn:
            model = algorithm.algorithm_class(**params)
            model.fit(features)
            return model.predict(features)

        elif sklearn:
            model = algorithm.algorithm_class(**params)
            return model.fit_predict(features)

        else:
            raise RuntimeError(f"Algorithm {algorithm.name} not available")

    def _evaluate_clustering_quality(self, labels: np.ndarray, features: np.ndarray) -> float:
        """Evaluate clustering quality using multiple metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score

            # Silhouette score
            silhouette = silhouette_score(features, labels)

            # Calinski-Harabasz score
            ch_score = calinski_harabasz_score(features, labels)

            # Combine scores
            combined_score = 0.6 * silhouette + 0.4 * ch_score

            return combined_score

        except Exception:
            # Fallback to simple score
            return len(np.unique(labels)) / len(labels)

    def _process_parallel_results(self, results: Dict[str, Any],
                                consensus_detector: EarlyConsensusDetector) -> Dict[str, Any]:
        """Process and aggregate parallel clustering results."""

        successful_results = {k: v for k, v in results.items() if 'error' not in v}

        if not successful_results:
            return {'error': 'No successful clustering results'}

        # Calculate algorithm weights based on performance
        algorithm_weights = {}
        for name, result in successful_results.items():
            algorithm_weights[name] = result.get('quality_score', 0.1)

        # Normalize weights
        total_weight = sum(algorithm_weights.values())
        if total_weight > 0:
            algorithm_weights = {k: v/total_weight for k, v in algorithm_weights.items()}

        # Get consensus result if early consensus was detected
        consensus_labels = None
        if self.early_consensus_detected:
            n_clusters = len(np.unique(list(successful_results.values())[0]['labels']))
            consensus_labels = consensus_detector.get_consensus_labels(n_clusters)

        return {
            'algorithm_results': successful_results,
            'algorithm_weights': algorithm_weights,
            'consensus_labels': consensus_labels,
            'early_consensus_detected': self.early_consensus_detected,
            'total_algorithms': len(results),
            'successful_algorithms': len(successful_results),
            'processing_stats': {
                'parallel_workers': self.max_workers,
                'completed_algorithms': self.completed_algorithms
            }
        }

class AdvancedEnsembleClustering:
    """Advanced ensemble clustering with hierarchical consensus and dynamic weighting."""
    @log_important_calls
    
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
    @log_all_calls
        
    def _initialize_clustering_algorithms(self):
        """Initialize comprehensive set of clustering algorithms."""
        self.algorithms = {}
        
        # HMM Clustering
        if hmmlearn:
            self.algorithms['hmm'] = ClusteringAlgorithm(
                name='hmm',
                algorithm_class = hmmlearn.hmm.GaussianHMM,
                parameter_ranges={
                    'n_components': (2, 8),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'n_iter': (50, 200),
                    'tol': (1e-6, 1e-2),
                    'reg_covar': (1e-7, 1e-2)
                },
                weight = 0.3,
                stability_threshold = 0.8
            )
        
        # K-Means variants
        if sklearn:
            from sklearn.cluster import KMeans, MiniBatchKMeans
            
            self.algorithms['kmeans'] = ClusteringAlgorithm(
                name='kmeans',
                algorithm_class = KMeans,
                parameter_ranges={
                    'n_clusters': (5, 30),
                    'n_init': (10, 20),
                    'max_iter': (100, 500),
                    'algorithm': ['lloyd', 'elkan']
                },
                weight = 0.2,
                stability_threshold = 0.7
            )
            
            self.algorithms['mini_batch_kmeans'] = ClusteringAlgorithm(
                name='mini_batch_kmeans',
                algorithm_class = MiniBatchKMeans,
                parameter_ranges={
                    'n_clusters': (5, 30),
                    'batch_size': (100, 1000),
                    'max_iter': (100, 500)
                },
                weight = 0.15,
                stability_threshold = 0.6
            )
        
        # Gaussian Mixture Models
        if sklearn:
            self.algorithms['gaussian_mixture'] = ClusteringAlgorithm(
                name='gaussian_mixture',
                algorithm_class = GaussianMixture,
                parameter_ranges={
                    'n_components': (2, 15),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'init_params': ['kmeans', 'random'],
                    'max_iter': (50, 200)
                },
                weight = 0.2,
                stability_threshold = 0.75
            )
            
            self.algorithms['bayesian_gaussian_mixture'] = ClusteringAlgorithm(
                name='bayesian_gaussian_mixture',
                algorithm_class = BayesianGaussianMixture,
                parameter_ranges={
                    'n_components': (2, 15),
                    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                    'init_params': ['kmeans', 'random'],
                    'max_iter': (50, 200)
                },
                weight = 0.25,
                stability_threshold = 0.8
            )
        
        # Density-based clustering
        if sklearn:
            from sklearn.cluster import DBSCAN
            
            self.algorithms['dbscan'] = ClusteringAlgorithm(
                name='dbscan',
                algorithm_class = DBSCAN,
                parameter_ranges={
                    'eps': (0.1, 2.0),
                    'min_samples': (5, 50),
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight = 0.15,
                stability_threshold = 0.6
            )
            
            self.algorithms['optics'] = ClusteringAlgorithm(
                name='optics',
                algorithm_class = OPTICS,
                parameter_ranges={
                    'min_samples': (5, 50),
                    'max_eps': (0.5, 5.0),
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight = 0.1,
                stability_threshold = 0.5
            )
        
        # Hierarchical clustering
        if sklearn:
            self.algorithms['agglomerative'] = ClusteringAlgorithm(
                name='agglomerative',
                algorithm_class = AgglomerativeClustering,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'linkage': ['ward', 'complete', 'average', 'single'],
                    'metric': ['euclidean', 'manhattan', 'cosine']
                },
                weight = 0.2,
                stability_threshold = 0.7
            )
            
            self.algorithms['birch'] = ClusteringAlgorithm(
                name='birch',
                algorithm_class = Birch,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'threshold': (0.1, 2.0),
                    'branching_factor': (10, 100)
                },
                weight = 0.15,
                stability_threshold = 0.6
            )
        
        # Spectral clustering
        if sklearn:
            self.algorithms['spectral'] = ClusteringAlgorithm(
                name='spectral',
                algorithm_class = SpectralClustering,
                parameter_ranges={
                    'n_clusters': (2, 20),
                    'affinity': ['rbf', 'nearest_neighbors', 'polynomial'],
                    'gamma': (0.1, 10.0),
                    'n_neighbors': (5, 50)
                },
                weight = 0.2,
                stability_threshold = 0.7
            )
    @log_all_calls
    
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
    @log_all_calls
    
    def _evaluate_clustering_stability(self, features: np.ndarray, algorithm: ClusteringAlgorithm, 
                                     params: Dict, n_bootstrap: int = 5) -> float:
        """Evaluate clustering stability using bootstrap sampling."""
        try:
            stability_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                n_samples = len(features)
                bootstrap_indices = np.random.choice(n_samples, size = n_samples, replace = True)
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
    @log_all_calls
    
    def _calculate_algorithm_weights(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate dynamic weights for each algorithm based on performance."""
        weights = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                # Generate parameter combinations
                param_combinations = self._generate_parameter_combinations(algorithm, n_combinations = 5)
                
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
    @log_all_calls
    
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
                param_combinations = self._generate_parameter_combinations(algorithm, n_combinations = 3)
                
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
    @log_all_calls
    
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
                        subset_indices = np.random.choice(n_samples, subset_size, replace = False)
                        
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
            
            return best_labels if best_labels is not None else np.zeros(len(consensus_matrix), dtype = int)
            
        except Exception as e:
            self.logger.error(f"Hierarchical consensus clustering failed: {e}")
            # Fallback to simple consensus voting
            return self._simple_consensus_voting(consensus_matrix)
    @log_all_calls
    
    def _simple_consensus_voting(self, consensus_matrix: np.ndarray) -> np.ndarray:
        """Simple consensus voting as fallback."""
        try:
            # Use spectral clustering on consensus matrix
            from sklearn.cluster import SpectralClustering
            
            # Estimate number of clusters
            n_clusters = min(8, max(2, len(consensus_matrix) // 100))
            
            spectral = SpectralClustering(
                n_clusters = n_clusters,
                affinity='precomputed',
                random_state = 42
            )
            
            labels = spectral.fit_predict(consensus_matrix)
            return labels
            
        except Exception as e:
            self.logger.error(f"Simple consensus voting failed: {e}")
            return np.zeros(len(consensus_matrix), dtype = int)
    @log_all_calls
    
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

    @log_important_calls
    def perform_parallel_ensemble_clustering(self, features: np.ndarray,
                                           data: pd.DataFrame = None,
                                           use_parallel: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform ensemble clustering with parallel processing and early consensus detection.

        This method provides significant performance improvements through:
        - Parallel algorithm execution
        - Early consensus detection to stop unnecessary computations
        - Memory-efficient processing

        Args:
            features: Feature matrix for clustering
            data: Original data (optional, used for evaluation)
            use_parallel: Whether to use parallel processing

        Returns:
            Tuple of (regime_labels, comprehensive_results)
        """

        self.logger.info("🚀 Starting enhanced parallel ensemble clustering")

        if use_parallel and JOBLIB_AVAILABLE:
            # Use parallel clustering processor
            self.logger.info("📊 Using parallel clustering with early consensus detection")

            # Initialize parallel processor
            parallel_config = ParallelClusteringConfig()
            parallel_processor = ParallelClusteringProcessor(parallel_config)

            # Process algorithms in parallel
            parallel_results = parallel_processor.process_algorithms_parallel(
                self.algorithms, features, data
            )

            if "error" in parallel_results:
                self.logger.warning(f"Parallel clustering failed: {parallel_results['error']}")
                # Fallback to sequential processing
                return self.perform_ensemble_clustering(features, data)

            # Process parallel results
            algorithm_results = parallel_results['algorithm_results']
            algorithm_weights = parallel_results['algorithm_weights']

            # Build consensus from parallel results
            if parallel_results['early_consensus_detected'] and parallel_results['consensus_labels'] is not None:
                # Use early consensus result
                self.final_regimes = parallel_results['consensus_labels']
                self.logger.info("✅ Using early consensus clustering result")
            else:
                # Build consensus from all results
                self.final_regimes, consensus_info = self._build_consensus_from_results(
                    algorithm_results, algorithm_weights
                )

            # Store results
            self.algorithm_weights = algorithm_weights
            self.clustering_results = algorithm_results

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(self.final_regimes, features)

            # Compile comprehensive results
            results = {
                'regimes': self.final_regimes,
                'consensus_matrix': None,  # Not available in parallel mode
                'algorithm_weights': self.algorithm_weights,
                'quality_metrics': quality_metrics,
                'n_regimes': len(np.unique(self.final_regimes)),
                'regime_distribution': {
                    f'regime_{regime}': int(np.sum(self.final_regimes == regime))
                    for regime in np.unique(self.final_regimes)
                },
                'parallel_processing': {
                    'parallel_workers': parallel_results['processing_stats']['parallel_workers'],
                    'total_algorithms': parallel_results['total_algorithms'],
                    'successful_algorithms': parallel_results['successful_algorithms'],
                    'early_consensus_detected': parallel_results['early_consensus_detected'],
                    'time_saved': parallel_results['processing_stats']['completed_algorithms'] <
                                parallel_results['total_algorithms']
                }
            }

            self.logger.info("✅ Enhanced parallel ensemble clustering completed")
            self.logger.info(f"   Discovered {results['n_regimes']} regimes")
            self.logger.info(f"   Quality score: {quality_metrics['quality_score']:.3f}")
            if parallel_results['early_consensus_detected']:
                self.logger.info("🎯 Early consensus optimization activated")

            return self.final_regimes, results

        else:
            # Fallback to sequential processing
            self.logger.info("📊 Using sequential ensemble clustering")
            return self.perform_ensemble_clustering(features, data)

    @log_all_calls
    def _build_consensus_from_results(self, algorithm_results: Dict[str, Any],
                                    algorithm_weights: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build consensus clustering from algorithm results."""

        if not algorithm_results:
            raise ValueError("No algorithm results available for consensus building")

        # Extract labels from all algorithms
        all_labels = []
        weights = []

        for algorithm_name, result in algorithm_results.items():
            if 'labels' in result:
                all_labels.append(result['labels'])
                weights.append(algorithm_weights.get(algorithm_name, 1.0))

        if not all_labels:
            raise ValueError("No valid labels found in algorithm results")

        # Convert to numpy array
        labels_matrix = np.column_stack(all_labels)
        weights = np.array(weights)

        # Build weighted consensus matrix
        n_samples = labels_matrix.shape[0]
        consensus_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                # Calculate weighted agreement
                agreements = [1.0 if labels_matrix[i, k] == labels_matrix[j, k] else 0.0
                            for k in range(len(weights))]
                consensus_matrix[i, j] = np.average(agreements, weights=weights)

        # Apply hierarchical clustering to consensus matrix
        from scipy.cluster.hierarchy import linkage, fcluster
        linkage_matrix = linkage(1 - consensus_matrix, method='average')

        # Determine optimal number of clusters
        n_clusters = self._determine_optimal_clusters(linkage_matrix, labels_matrix)

        consensus_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        consensus_info = {
            'n_clusters': n_clusters,
            'consensus_matrix_shape': consensus_matrix.shape,
            'algorithm_contributions': len(algorithm_results)
        }

        return consensus_labels, consensus_info

    @log_all_calls
    def _determine_optimal_clusters(self, linkage_matrix: np.ndarray,
                                  labels_matrix: np.ndarray) -> int:
        """Determine optimal number of clusters using multiple criteria."""

        # Try different numbers of clusters
        max_clusters = min(20, linkage_matrix.shape[0] // 10)
        cluster_range = range(2, max_clusters + 1)

        best_score = -float('inf')
        best_n_clusters = 2

        for n_clusters in cluster_range:
            try:
                # Get cluster labels
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                # Calculate stability score
                stability_score = self._calculate_cluster_stability(labels, labels_matrix)

                if stability_score > best_score:
                    best_score = stability_score
                    best_n_clusters = n_clusters

            except Exception:
                continue

        return best_n_clusters

    @log_all_calls
    def _calculate_cluster_stability(self, consensus_labels: np.ndarray,
                                   algorithm_labels: np.ndarray) -> float:
        """Calculate stability of consensus clustering."""

        stability_scores = []

        for i in range(algorithm_labels.shape[1]):
            # Calculate agreement between consensus and individual algorithm
            from sklearn.metrics import adjusted_rand_score
            ari_score = adjusted_rand_score(consensus_labels, algorithm_labels[:, i])
            stability_scores.append(ari_score)

        return np.mean(stability_scores) if stability_scores else 0.0