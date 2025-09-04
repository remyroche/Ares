#!/usr/bin/env python3
"""Ensemble Clustering Methods for HMM Regime Discovery - Vectorized Implementation.

This module implements computationally efficient ensemble clustering using vectorized
operations for HMM + K-means + DBSCAN combination.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

class EnsembleClusteringRegimeDetector:
    """Vectorized ensemble clustering for regime detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_jobs = self.config.get('n_jobs', -1)
        self.random_state = self.config.get('random_state', 42)
        
        # Clustering parameters
        self.hmm_params = self.config.get('hmm_params', {
            'n_components_range': [2, 8],
            'covariance_types': ['full', 'tied', 'diag', 'spherical']
        })
        
        self.kmeans_params = self.config.get('kmeans_params', {
            'n_clusters_range': [10, 30],
            'n_init': 10,
            'max_iter': 300
        })
        
        self.dbscan_params = self.config.get('dbscan_params', {
            'eps_range': [0.1, 2.0],
            'min_samples_range': [5, 50]
        })
        
        # Ensemble weights
        self.ensemble_weights = self.config.get('ensemble_weights', {
            'hmm': 0.4,
            'kmeans': 0.3,
            'dbscan': 0.3
        })
    
    def ensemble_regime_detection(self, features: np.ndarray, method_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform optimized ensemble regime detection using HMM + K-means + DBSCAN.
        
        Args:
            features: Input features for clustering
            method_weights: Optional weights for ensemble methods
            
        Returns:
            Tuple of (consensus_regimes, ensemble_results)
        """
        if method_weights is None:
            method_weights = self.ensemble_weights
        
        # 1. Optimize feature preprocessing
        features_optimized = self._optimize_feature_preprocessing(features)
        
        # 2. Parallel clustering execution
        ensemble_results = self._parallel_clustering_execution(features_optimized)
        
        # 3. Fast quality assessment
        quality_weights = self._fast_quality_assessment(ensemble_results, features_optimized)
        
        # 4. Optimized consensus voting
        consensus_regimes = self._optimized_consensus_voting(
            ensemble_results, quality_weights, method_weights
        )
        
        # 5. Vectorized confidence calculation
        confidence_scores = self._vectorized_confidence_calculation(
            ensemble_results, consensus_regimes
        )
        
        # 6. Compile final results
        final_results = {
            'consensus_regimes': consensus_regimes,
            'confidence_scores': confidence_scores,
            'ensemble_results': ensemble_results,
            'quality_weights': quality_weights,
            'method_weights': method_weights,
            'n_regimes': len(np.unique(consensus_regimes)),
            'ensemble_quality': self._fast_ensemble_quality_assessment(features_optimized, consensus_regimes)
        }
        
        return consensus_regimes, final_results
    
    def _optimize_feature_preprocessing(self, features: np.ndarray) -> np.ndarray:
        """Optimize feature preprocessing for efficiency."""
        # Use incremental PCA for dimensionality reduction if needed
        if features.shape[1] > 50:
            from sklearn.decomposition import IncrementalPCA
            n_components = min(50, features.shape[1] // 2)
            pca = IncrementalPCA(n_components=n_components, batch_size=1000)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        # Use robust scaling for better outlier handling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_reduced)
        
        return features_scaled
    
    def _parallel_clustering_execution(self, features: np.ndarray) -> Dict[str, Any]:
        """Execute clustering methods in parallel for efficiency."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        ensemble_results = {}
        
        # Define clustering tasks
        tasks = {
            'hmm': lambda: self._hmm_clustering_optimized(features),
            'kmeans': lambda: self._kmeans_clustering_optimized(features),
            'dbscan': lambda: self._dbscan_clustering_optimized(features)
        }
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_method = {executor.submit(task): method for method, task in tasks.items()}
            
            for future in as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    result = future.result()
                    ensemble_results[method] = result
                except Exception as e:
                    print(f"Error in {method} clustering: {e}")
                    ensemble_results[method] = {'regimes': np.zeros(len(features)), 'quality_score': 0.0}
        
        return ensemble_results
    
    def _hmm_clustering_optimized(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimized HMM clustering with reduced parameter space."""
        try:
            from hmmlearn import hmm
        except ImportError:
            return {'regimes': np.zeros(len(features)), 'quality_score': 0.0, 'model': None}
        
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Reduced parameter space for efficiency
        n_components_range = [3, 4, 5]  # Reduced from [2, 8]
        covariance_types = ['full', 'tied']  # Reduced from 4 types
        
        for n_components in n_components_range:
            for covariance_type in covariance_types:
                try:
                    # Use subset for faster training
                    max_samples = min(5000, len(features))
                    if len(features) > max_samples:
                        indices = np.random.choice(len(features), max_samples, replace=False)
                        features_subset = features[indices]
                    else:
                        features_subset = features
                    
                    # Train HMM with reduced iterations
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=50,  # Reduced from 100
                        random_state=self.random_state
                    )
                    
                    model.fit(features_subset)
                    
                    # Get regime predictions
                    regimes = model.predict(features)
                    regime_probs = model.predict_proba(features)
                    
                    # Fast quality score calculation
                    quality_score = self._fast_hmm_quality_score(features, regimes, regime_probs)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_regimes = regimes
                        best_model = model
                        best_params = {'n_components': n_components, 'covariance_type': covariance_type}
                
                except Exception as e:
                    continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _kmeans_clustering_optimized(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimized K-means clustering with reduced parameter space."""
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Reduced parameter space
        n_clusters_range = [15, 20, 25]  # Reduced from [10, 30]
        
        for n_clusters in n_clusters_range:
            try:
                # Train K-means with reduced parameters
                model = KMeans(
                    n_clusters=n_clusters,
                    n_init=5,  # Reduced from 10
                    max_iter=100,  # Reduced from 300
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                
                regimes = model.fit_predict(features)
                
                # Fast quality score calculation
                quality_score = self._fast_kmeans_quality_score(features, regimes, model)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_regimes = regimes
                    best_model = model
                    best_params = {'n_clusters': n_clusters}
            
            except Exception as e:
                continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _dbscan_clustering_optimized(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimized DBSCAN clustering with reduced parameter space."""
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Reduced parameter space
        eps_values = [0.3, 0.5, 0.7]  # Reduced from 10 values
        min_samples_values = [5, 10, 15]  # Reduced from range
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Train DBSCAN
                    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
                    regimes = model.fit_predict(features)
                    
                    # Skip if too many noise points or too few clusters
                    n_noise = np.sum(regimes == -1)
                    n_clusters = len(np.unique(regimes)) - (1 if n_noise > 0 else 0)
                    
                    if n_noise > len(features) * 0.5 or n_clusters < 2:
                        continue
                    
                    # Fast quality score calculation
                    quality_score = self._fast_dbscan_quality_score(features, regimes, model)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_regimes = regimes
                        best_model = model
                        best_params = {'eps': eps, 'min_samples': min_samples}
                
                except Exception as e:
                    continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _fast_hmm_quality_score(self, features: np.ndarray, regimes: np.ndarray, regime_probs: np.ndarray) -> float:
        """Fast HMM quality score calculation."""
        try:
            unique_regimes = np.unique(regimes)
            if len(unique_regimes) < 2:
                return 0.0
            
            # Fast regime balance calculation
            regime_counts = np.bincount(regimes)
            balance_score = 1.0 - (np.std(regime_counts) / np.mean(regime_counts))
            
            # Fast stability score
            max_probs = np.max(regime_probs, axis=1)
            stability_score = np.mean(max_probs)
            
            # Combined score
            quality_score = 0.6 * balance_score + 0.4 * stability_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _fast_kmeans_quality_score(self, features: np.ndarray, regimes: np.ndarray, model: KMeans) -> float:
        """Fast K-means quality score calculation."""
        try:
            # Fast silhouette score calculation using subset
            sample_size = min(1000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[sample_indices]
            regimes_sample = regimes[sample_indices]
            
            if len(np.unique(regimes_sample)) > 1:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(features_sample, regimes_sample)
            else:
                silhouette = 0.0
            
            # Fast inertia score
            inertia_score = 1 / (1 + model.inertia_ / len(features))
            
            # Combined score
            quality_score = 0.7 * silhouette + 0.3 * inertia_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _fast_dbscan_quality_score(self, features: np.ndarray, regimes: np.ndarray, model: DBSCAN) -> float:
        """Fast DBSCAN quality score calculation."""
        try:
            # Remove noise points for quality calculation
            non_noise_mask = regimes != -1
            if np.sum(non_noise_mask) < 10:
                return 0.0
            
            features_clean = features[non_noise_mask]
            regimes_clean = regimes[non_noise_mask]
            
            # Fast silhouette score using subset
            sample_size = min(1000, len(features_clean))
            sample_indices = np.random.choice(len(features_clean), sample_size, replace=False)
            features_sample = features_clean[sample_indices]
            regimes_sample = regimes_clean[sample_indices]
            
            if len(np.unique(regimes_sample)) > 1:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(features_sample, regimes_sample)
            else:
                silhouette = 0.0
            
            # Noise ratio score
            noise_ratio = np.sum(regimes == -1) / len(regimes)
            noise_score = 1 - noise_ratio
            
            # Combined score
            quality_score = 0.7 * silhouette + 0.3 * noise_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _fast_quality_assessment(self, ensemble_results: Dict[str, Any], features: np.ndarray) -> Dict[str, float]:
        """Fast quality assessment for ensemble methods."""
        quality_scores = {}
        for method, results in ensemble_results.items():
            quality_scores[method] = results.get('quality_score', 0.0)
        
        # Normalize weights
        total_score = sum(quality_scores.values())
        if total_score > 0:
            quality_weights = {method: score / total_score for method, score in quality_scores.items()}
        else:
            quality_weights = {method: 1.0 / len(quality_scores) for method in quality_scores.keys()}
        
        return quality_weights
    
    def _optimized_consensus_voting(self, ensemble_results: Dict[str, Any], 
                                  quality_weights: Dict[str, float],
                                  method_weights: Dict[str, float]) -> np.ndarray:
        """Optimized consensus voting with vectorized operations."""
        n_samples = len(list(ensemble_results.values())[0]['regimes'])
        n_methods = len(ensemble_results)
        
        # Create regime matrix efficiently
        regime_matrix = np.zeros((n_samples, n_methods), dtype=int)
        weight_matrix = np.zeros((n_samples, n_methods), dtype=float)
        
        method_names = list(ensemble_results.keys())
        for i, method in enumerate(method_names):
            regime_matrix[:, i] = ensemble_results[method]['regimes']
            combined_weight = quality_weights[method] * method_weights[method]
            weight_matrix[:, i] = combined_weight
        
        # Vectorized consensus voting
        consensus_regimes = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            sample_regimes = regime_matrix[i, :]
            sample_weights = weight_matrix[i, :]
            
            # Calculate weighted votes for each unique regime
            unique_regimes = np.unique(sample_regimes)
            regime_votes = {}
            
            for regime in unique_regimes:
                regime_mask = sample_regimes == regime
                regime_vote = np.sum(sample_weights[regime_mask])
                regime_votes[regime] = regime_vote
            
            # Assign regime with highest vote
            consensus_regimes[i] = max(regime_votes, key=regime_votes.get)
        
        return consensus_regimes
    
    def _vectorized_confidence_calculation(self, ensemble_results: Dict[str, Any], 
                                         consensus_regimes: np.ndarray) -> np.ndarray:
        """Vectorized confidence calculation."""
        n_samples = len(consensus_regimes)
        n_methods = len(ensemble_results)
        
        # Create agreement matrix
        agreement_matrix = np.zeros((n_samples, n_methods), dtype=bool)
        
        method_names = list(ensemble_results.keys())
        for i, method in enumerate(method_names):
            method_regimes = ensemble_results[method]['regimes']
            agreement_matrix[:, i] = (method_regimes == consensus_regimes)
        
        # Calculate confidence as agreement ratio
        confidence_scores = np.mean(agreement_matrix, axis=1)
        
        return confidence_scores
    
    def _fast_ensemble_quality_assessment(self, features: np.ndarray, consensus_regimes: np.ndarray) -> Dict[str, float]:
        """Fast ensemble quality assessment."""
        try:
            unique_regimes = np.unique(consensus_regimes)
            n_regimes = len(unique_regimes)
            
            if n_regimes < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
            
            # Use subset for fast calculation
            sample_size = min(2000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[sample_indices]
            regimes_sample = consensus_regimes[sample_indices]
            
            # Calculate clustering quality metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            silhouette = silhouette_score(features_sample, regimes_sample)
            calinski_harabasz = calinski_harabasz_score(features_sample, regimes_sample)
            davies_bouldin = davies_bouldin_score(features_sample, regimes_sample)
            
            # Calculate regime balance
            regime_sizes = [np.sum(consensus_regimes == regime) for regime in unique_regimes]
            regime_balance = 1 / (1 + np.std(regime_sizes) / np.mean(regime_sizes))
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'regime_balance': regime_balance,
                'n_regimes': n_regimes
            }
            
        except Exception as e:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
    
    def _hmm_clustering_vectorized(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform HMM clustering with vectorized operations."""
        try:
            from hmmlearn import hmm
        except ImportError:
            return {'regimes': np.zeros(len(features)), 'quality_score': 0.0, 'model': None}
        
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Grid search over HMM parameters
        for n_components in range(self.hmm_params['n_components_range'][0], 
                                 self.hmm_params['n_components_range'][1] + 1):
            for covariance_type in self.hmm_params['covariance_types']:
                try:
                    # Train HMM
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=100,
                        random_state=self.random_state
                    )
                    
                    # Use subset for faster training
                    max_samples = min(10000, len(features))
                    if len(features) > max_samples:
                        indices = np.random.choice(len(features), max_samples, replace=False)
                        features_subset = features[indices]
                    else:
                        features_subset = features
                    
                    model.fit(features_subset)
                    
                    # Get regime predictions
                    regimes = model.predict(features)
                    regime_probs = model.predict_proba(features)
                    
                    # Calculate quality score
                    quality_score = self._calculate_hmm_quality_score(features, regimes, regime_probs)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_regimes = regimes
                        best_model = model
                        best_params = {'n_components': n_components, 'covariance_type': covariance_type}
                
                except Exception as e:
                    continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _kmeans_clustering_vectorized(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform K-means clustering with vectorized operations."""
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Grid search over K-means parameters
        for n_clusters in range(self.kmeans_params['n_clusters_range'][0], 
                               self.kmeans_params['n_clusters_range'][1] + 1):
            try:
                # Train K-means
                model = KMeans(
                    n_clusters=n_clusters,
                    n_init=self.kmeans_params['n_init'],
                    max_iter=self.kmeans_params['max_iter'],
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                
                regimes = model.fit_predict(features)
                
                # Calculate quality score
                quality_score = self._calculate_kmeans_quality_score(features, regimes, model)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_regimes = regimes
                    best_model = model
                    best_params = {'n_clusters': n_clusters}
            
            except Exception as e:
                continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _dbscan_clustering_vectorized(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform DBSCAN clustering with vectorized operations."""
        best_score = -np.inf
        best_regimes = None
        best_model = None
        best_params = None
        
        # Grid search over DBSCAN parameters
        eps_values = np.linspace(self.dbscan_params['eps_range'][0], 
                                self.dbscan_params['eps_range'][1], 10)
        min_samples_values = range(self.dbscan_params['min_samples_range'][0], 
                                  self.dbscan_params['min_samples_range'][1] + 1, 5)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Train DBSCAN
                    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.n_jobs)
                    regimes = model.fit_predict(features)
                    
                    # Skip if too many noise points or too few clusters
                    n_noise = np.sum(regimes == -1)
                    n_clusters = len(np.unique(regimes)) - (1 if n_noise > 0 else 0)
                    
                    if n_noise > len(features) * 0.5 or n_clusters < 2:
                        continue
                    
                    # Calculate quality score
                    quality_score = self._calculate_dbscan_quality_score(features, regimes, model)
                    
                    if quality_score > best_score:
                        best_score = quality_score
                        best_regimes = regimes
                        best_model = model
                        best_params = {'eps': eps, 'min_samples': min_samples}
                
                except Exception as e:
                    continue
        
        return {
            'regimes': best_regimes if best_regimes is not None else np.zeros(len(features)),
            'quality_score': best_score,
            'model': best_model,
            'params': best_params
        }
    
    def _calculate_hmm_quality_score(self, features: np.ndarray, regimes: np.ndarray, regime_probs: np.ndarray) -> float:
        """Calculate quality score for HMM clustering."""
        try:
            # 1. Regime separation score
            unique_regimes = np.unique(regimes)
            if len(unique_regimes) < 2:
                return 0.0
            
            # Calculate regime centroids
            regime_centroids = []
            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 0:
                    centroid = np.mean(features[regime_mask], axis=0)
                    regime_centroids.append(centroid)
            
            if len(regime_centroids) < 2:
                return 0.0
            
            # Calculate inter-regime distances
            regime_centroids = np.array(regime_centroids)
            inter_distances = pdist(regime_centroids)
            separation_score = np.mean(inter_distances)
            
            # 2. Regime stability score (based on probability consistency)
            max_probs = np.max(regime_probs, axis=1)
            stability_score = np.mean(max_probs)
            
            # 3. Regime balance score
            regime_sizes = [np.sum(regimes == regime) for regime in unique_regimes]
            balance_score = 1 / (1 + np.std(regime_sizes) / np.mean(regime_sizes))
            
            # Combined score
            quality_score = 0.4 * separation_score + 0.4 * stability_score + 0.2 * balance_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _calculate_kmeans_quality_score(self, features: np.ndarray, regimes: np.ndarray, model: KMeans) -> float:
        """Calculate quality score for K-means clustering."""
        try:
            # 1. Silhouette score
            if len(np.unique(regimes)) > 1:
                silhouette = silhouette_score(features, regimes)
            else:
                silhouette = 0.0
            
            # 2. Inertia (inverse for higher is better)
            inertia_score = 1 / (1 + model.inertia_ / len(features))
            
            # 3. Regime balance score
            unique_regimes = np.unique(regimes)
            regime_sizes = [np.sum(regimes == regime) for regime in unique_regimes]
            balance_score = 1 / (1 + np.std(regime_sizes) / np.mean(regime_sizes))
            
            # Combined score
            quality_score = 0.5 * silhouette + 0.3 * inertia_score + 0.2 * balance_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _calculate_dbscan_quality_score(self, features: np.ndarray, regimes: np.ndarray, model: DBSCAN) -> float:
        """Calculate quality score for DBSCAN clustering."""
        try:
            # Remove noise points for quality calculation
            non_noise_mask = regimes != -1
            if np.sum(non_noise_mask) < 10:
                return 0.0
            
            features_clean = features[non_noise_mask]
            regimes_clean = regimes[non_noise_mask]
            
            # 1. Silhouette score
            if len(np.unique(regimes_clean)) > 1:
                silhouette = silhouette_score(features_clean, regimes_clean)
            else:
                silhouette = 0.0
            
            # 2. Noise ratio (lower is better)
            noise_ratio = np.sum(regimes == -1) / len(regimes)
            noise_score = 1 - noise_ratio
            
            # 3. Number of clusters (prefer moderate number)
            n_clusters = len(np.unique(regimes_clean))
            cluster_score = 1 / (1 + abs(n_clusters - 10) / 10)  # Prefer around 10 clusters
            
            # Combined score
            quality_score = 0.5 * silhouette + 0.3 * noise_score + 0.2 * cluster_score
            
            return quality_score
            
        except Exception as e:
            return 0.0
    
    def _calculate_quality_weights(self, ensemble_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights based on quality scores."""
        quality_scores = {}
        for method, results in ensemble_results.items():
            quality_scores[method] = results.get('quality_score', 0.0)
        
        # Normalize weights
        total_score = sum(quality_scores.values())
        if total_score > 0:
            quality_weights = {method: score / total_score for method, score in quality_scores.items()}
        else:
            # Equal weights if all scores are 0
            quality_weights = {method: 1.0 / len(quality_scores) for method in quality_scores.keys()}
        
        return quality_weights
    
    def _weighted_consensus_voting_vectorized(self, ensemble_results: Dict[str, Any], 
                                            quality_weights: Dict[str, float],
                                            method_weights: Dict[str, float]) -> np.ndarray:
        """Perform weighted consensus voting using vectorized operations."""
        n_samples = len(list(ensemble_results.values())[0]['regimes'])
        n_methods = len(ensemble_results)
        
        # Create regime matrix
        regime_matrix = np.zeros((n_samples, n_methods), dtype=int)
        weight_matrix = np.zeros((n_samples, n_methods), dtype=float)
        
        method_names = list(ensemble_results.keys())
        for i, method in enumerate(method_names):
            regime_matrix[:, i] = ensemble_results[method]['regimes']
            # Combine quality and method weights
            combined_weight = quality_weights[method] * method_weights[method]
            weight_matrix[:, i] = combined_weight
        
        # Vectorized consensus voting
        consensus_regimes = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Get regime votes for this sample
            sample_regimes = regime_matrix[i, :]
            sample_weights = weight_matrix[i, :]
            
            # Calculate weighted votes for each unique regime
            unique_regimes = np.unique(sample_regimes)
            regime_votes = {}
            
            for regime in unique_regimes:
                regime_mask = sample_regimes == regime
                regime_vote = np.sum(sample_weights[regime_mask])
                regime_votes[regime] = regime_vote
            
            # Assign regime with highest vote
            consensus_regimes[i] = max(regime_votes, key=regime_votes.get)
        
        return consensus_regimes
    
    def _calculate_ensemble_confidence_vectorized(self, ensemble_results: Dict[str, Any], 
                                                consensus_regimes: np.ndarray) -> np.ndarray:
        """Calculate ensemble confidence scores using vectorized operations."""
        n_samples = len(consensus_regimes)
        n_methods = len(ensemble_results)
        
        # Create agreement matrix
        agreement_matrix = np.zeros((n_samples, n_methods), dtype=bool)
        
        method_names = list(ensemble_results.keys())
        for i, method in enumerate(method_names):
            method_regimes = ensemble_results[method]['regimes']
            agreement_matrix[:, i] = (method_regimes == consensus_regimes)
        
        # Calculate confidence as agreement ratio
        confidence_scores = np.mean(agreement_matrix, axis=1)
        
        return confidence_scores
    
    def _calculate_ensemble_quality(self, features: np.ndarray, consensus_regimes: np.ndarray) -> Dict[str, float]:
        """Calculate overall ensemble quality metrics."""
        try:
            unique_regimes = np.unique(consensus_regimes)
            n_regimes = len(unique_regimes)
            
            if n_regimes < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
            
            # Calculate clustering quality metrics
            silhouette = silhouette_score(features, consensus_regimes)
            calinski_harabasz = calinski_harabasz_score(features, consensus_regimes)
            davies_bouldin = davies_bouldin_score(features, consensus_regimes)
            
            # Calculate regime balance
            regime_sizes = [np.sum(consensus_regimes == regime) for regime in unique_regimes]
            regime_balance = 1 / (1 + np.std(regime_sizes) / np.mean(regime_sizes))
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'regime_balance': regime_balance,
                'n_regimes': n_regimes
            }
            
        except Exception as e:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0, 'davies_bouldin_score': float('inf')}
    
    def optimize_ensemble_parameters(self, features: np.ndarray, 
                                   target_n_regimes: Optional[int] = None) -> Dict[str, Any]:
        """Optimize ensemble parameters based on data characteristics."""
        # Analyze data characteristics
        n_samples, n_features = features.shape
        
        # Adjust parameters based on data size
        if n_samples < 1000:
            # Small dataset - reduce parameter ranges
            self.hmm_params['n_components_range'] = [2, 4]
            self.kmeans_params['n_clusters_range'] = [5, 15]
            self.dbscan_params['eps_range'] = [0.5, 1.5]
            self.dbscan_params['min_samples_range'] = [3, 10]
        elif n_samples > 10000:
            # Large dataset - increase parameter ranges
            self.hmm_params['n_components_range'] = [3, 10]
            self.kmeans_params['n_clusters_range'] = [15, 50]
            self.dbscan_params['eps_range'] = [0.1, 3.0]
            self.dbscan_params['min_samples_range'] = [10, 100]
        
        # Adjust based on target number of regimes
        if target_n_regimes is not None:
            self.kmeans_params['n_clusters_range'] = [
                max(5, target_n_regimes - 5),
                min(50, target_n_regimes + 10)
            ]
        
        return {
            'hmm_params': self.hmm_params,
            'kmeans_params': self.kmeans_params,
            'dbscan_params': self.dbscan_params
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    # Create features with some structure
    features = np.random.randn(n_samples, n_features)
    
    # Add some regime structure
    regime1_mask = np.arange(n_samples) < n_samples // 3
    regime2_mask = (np.arange(n_samples) >= n_samples // 3) & (np.arange(n_samples) < 2 * n_samples // 3)
    regime3_mask = np.arange(n_samples) >= 2 * n_samples // 3
    
    features[regime1_mask] += np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    features[regime2_mask] += np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    features[regime3_mask] += np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Initialize ensemble detector
    config = {
        'n_jobs': -1,
        'random_state': 42,
        'ensemble_weights': {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3}
    }
    
    detector = EnsembleClusteringRegimeDetector(config)
    
    # Run ensemble clustering
    consensus_regimes, results = detector.ensemble_regime_detection(features)
    
    print(f"Ensemble Clustering Results:")
    print(f"Number of regimes: {results['n_regimes']}")
    print(f"Ensemble quality: {results['ensemble_quality']}")
    print(f"Quality weights: {results['quality_weights']}")
    print(f"Method weights: {results['method_weights']}")
    
    # Display individual method results
    print("\nIndividual Method Results:")
    for method, method_results in results['ensemble_results'].items():
        print(f"{method.upper()}:")
        print(f"  Quality Score: {method_results['quality_score']:.4f}")
        print(f"  Parameters: {method_results['params']}")
        print(f"  Unique Regimes: {len(np.unique(method_results['regimes']))}")
    
    # Display consensus results
    print(f"\nConsensus Results:")
    print(f"Unique Regimes: {np.unique(consensus_regimes)}")
    print(f"Regime Distribution: {np.bincount(consensus_regimes)}")
    print(f"Mean Confidence: {np.mean(results['confidence_scores']):.4f}")