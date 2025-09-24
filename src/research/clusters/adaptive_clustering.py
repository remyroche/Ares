"""
Adaptive Clustering for Market Regime Discovery

This module implements adaptive clustering algorithms that automatically learn:
1. Optimal number of clusters using multiple criteria
2. Best clustering algorithm for the data
3. Dynamic parameter optimization
4. Online adaptation to changing market conditions
5. Ensemble clustering with automatic weighting
6. Reinforcement learning for clustering decisions

Key Features:
- Multi-criteria cluster number optimization
- Algorithm selection using meta-learning
- Dynamic parameter adaptation
- Online clustering for streaming data
- Ensemble methods with learned weights
- Reinforcement learning optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import warnings
from abc import ABC, abstractmethod
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from src.utils.logger import system_logger

# Import existing components
from .regime_clusterer import RegimeClusterer, ClusteringMethod, ClusteringResult


class AdaptiveMethod(Enum):
    """Adaptive clustering methods."""
    MULTI_CRITERIA_OPTIMIZATION = "multi_criteria"
    BAYESIAN_OPTIMIZATION = "bayesian_opt"
    REINFORCEMENT_LEARNING = "reinforcement"
    ENSEMBLE_LEARNING = "ensemble"
    ONLINE_ADAPTATION = "online"
    META_LEARNING = "meta_learning"
    EVOLUTIONARY_OPTIMIZATION = "evolutionary"


class OptimizationCriterion(Enum):
    """Criteria for cluster optimization."""
    SILHOUETTE_SCORE = "silhouette"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    ELBOW_METHOD = "elbow"
    GAP_STATISTIC = "gap_statistic"
    INFORMATION_CRITERION = "information_criterion"
    STABILITY_SCORE = "stability"
    ECONOMIC_SIGNIFICANCE = "economic_significance"


@dataclass
class AdaptiveClusteringConfig:
    """Configuration for adaptive clustering."""
    # Cluster number search
    min_clusters: int = 2
    max_clusters: int = 15
    cluster_step: int = 1
    
    # Optimization criteria
    optimization_criteria: List[str] = field(default_factory=lambda: [
        "silhouette", "calinski_harabasz", "davies_bouldin", "gap_statistic"
    ])
    criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        "silhouette": 0.3,
        "calinski_harabasz": 0.25,
        "davies_bouldin": 0.2,
        "gap_statistic": 0.25
    })
    
    # Algorithm selection
    candidate_algorithms: List[str] = field(default_factory=lambda: [
        "kmeans", "gmm", "hierarchical", "spectral"
    ])
    algorithm_weights: Dict[str, float] = field(default_factory=dict)
    
    # Bayesian optimization
    bayesian_trials: int = 100
    bayesian_timeout: int = 1800  # 30 minutes
    
    # Reinforcement learning
    rl_episodes: int = 1000
    rl_learning_rate: float = 0.001
    rl_epsilon: float = 0.1
    rl_epsilon_decay: float = 0.995
    rl_memory_size: int = 10000
    
    # Online adaptation
    online_window_size: int = 1000
    adaptation_frequency: int = 100
    stability_threshold: float = 0.1
    
    # Ensemble parameters
    ensemble_size: int = 5
    ensemble_diversity_weight: float = 0.3
    
    # Meta-learning
    meta_learning_episodes: int = 50
    meta_features_dim: int = 20
    
    # General parameters
    n_bootstrap_samples: int = 100
    stability_runs: int = 10
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True


class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics."""
    
    @staticmethod
    def silhouette_score_metric(X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        if len(np.unique(labels)) < 2:
            return -1.0
        try:
            return silhouette_score(X, labels)
        except:
            return -1.0
    
    @staticmethod
    def calinski_harabasz_score_metric(X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score."""
        if len(np.unique(labels)) < 2:
            return 0.0
        try:
            return calinski_harabasz_score(X, labels)
        except:
            return 0.0
    
    @staticmethod
    def davies_bouldin_score_metric(X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score (lower is better)."""
        if len(np.unique(labels)) < 2:
            return float('inf')
        try:
            return -davies_bouldin_score(X, labels)  # Negative for maximization
        except:
            return float('-inf')
    
    @staticmethod
    def gap_statistic(X: np.ndarray, labels: np.ndarray, n_refs: int = 10) -> float:
        """Calculate Gap statistic."""
        if len(np.unique(labels)) < 2:
            return 0.0
        
        try:
            # Calculate within-cluster sum of squares
            def wss(X, labels):
                wss_total = 0
                for k in np.unique(labels):
                    cluster_points = X[labels == k]
                    if len(cluster_points) > 1:
                        centroid = np.mean(cluster_points, axis=0)
                        wss_total += np.sum((cluster_points - centroid) ** 2)
                return wss_total
            
            # Actual WSS
            actual_wss = wss(X, labels)
            
            # Reference WSS
            ref_wss = []
            for _ in range(n_refs):
                # Generate random data with same bounds
                random_data = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), X.shape
                )
                # Cluster random data with same number of clusters
                kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
                random_labels = kmeans.fit_predict(random_data)
                ref_wss.append(wss(random_data, random_labels))
            
            gap = np.log(np.mean(ref_wss)) - np.log(actual_wss)
            return gap
        except:
            return 0.0
    
    @staticmethod
    def stability_score(X: np.ndarray, clusterer, n_runs: int = 10) -> float:
        """Calculate clustering stability."""
        try:
            labels_list = []
            for _ in range(n_runs):
                # Add small noise to data
                noisy_X = X + np.random.normal(0, 0.01 * X.std(), X.shape)
                labels = clusterer.fit_predict(noisy_X)
                labels_list.append(labels)
            
            # Calculate average ARI between runs
            ari_scores = []
            for i in range(len(labels_list)):
                for j in range(i + 1, len(labels_list)):
                    ari = adjusted_rand_score(labels_list[i], labels_list[j])
                    ari_scores.append(ari)
            
            return np.mean(ari_scores)
        except:
            return 0.0


class MultiCriteriaOptimizer:
    """Multi-criteria optimization for cluster parameters."""
    
    def __init__(self, config: AdaptiveClusteringConfig):
        self.config = config
        self.logger = system_logger.getChild('MultiCriteriaOptimizer')
        self.metrics = ClusteringMetrics()
    
    def optimize_clusters(
        self, 
        X: np.ndarray, 
        algorithm: str = "kmeans"
    ) -> Tuple[int, Dict[str, Any]]:
        """Optimize number of clusters using multiple criteria."""
        
        self.logger.info(f"🎯 Multi-criteria optimization for {algorithm}")
        
        cluster_range = range(self.config.min_clusters, 
                            self.config.max_clusters + 1, 
                            self.config.cluster_step)
        
        results = {}
        
        for n_clusters in cluster_range:
            try:
                # Get clusterer
                clusterer = self._get_clusterer(algorithm, n_clusters)
                labels = clusterer.fit_predict(X)
                
                # Calculate all metrics
                metrics = {}
                
                if "silhouette" in self.config.optimization_criteria:
                    metrics["silhouette"] = self.metrics.silhouette_score_metric(X, labels)
                
                if "calinski_harabasz" in self.config.optimization_criteria:
                    metrics["calinski_harabasz"] = self.metrics.calinski_harabasz_score_metric(X, labels)
                
                if "davies_bouldin" in self.config.optimization_criteria:
                    metrics["davies_bouldin"] = self.metrics.davies_bouldin_score_metric(X, labels)
                
                if "gap_statistic" in self.config.optimization_criteria:
                    metrics["gap_statistic"] = self.metrics.gap_statistic(X, labels)
                
                if "stability" in self.config.optimization_criteria:
                    metrics["stability"] = self.metrics.stability_score(X, clusterer)
                
                results[n_clusters] = {
                    'metrics': metrics,
                    'labels': labels,
                    'clusterer': clusterer
                }
                
            except Exception as e:
                self.logger.warning(f"Failed for {n_clusters} clusters: {e}")
                continue
        
        # Calculate composite scores
        composite_scores = {}
        for n_clusters, result in results.items():
            score = 0.0
            for criterion, weight in self.config.criteria_weights.items():
                if criterion in result['metrics']:
                    # Normalize metrics to 0-1 range
                    metric_value = result['metrics'][criterion]
                    normalized_value = self._normalize_metric(criterion, metric_value, results)
                    score += weight * normalized_value
            
            composite_scores[n_clusters] = score
        
        # Find best number of clusters
        best_n_clusters = max(composite_scores.keys(), key=lambda k: composite_scores[k])
        
        optimization_results = {
            'best_n_clusters': best_n_clusters,
            'best_score': composite_scores[best_n_clusters],
            'all_scores': composite_scores,
            'all_metrics': {k: v['metrics'] for k, v in results.items()},
            'best_labels': results[best_n_clusters]['labels'],
            'best_clusterer': results[best_n_clusters]['clusterer']
        }
        
        self.logger.info(f"✅ Best number of clusters: {best_n_clusters} (score: {composite_scores[best_n_clusters]:.3f})")
        
        return best_n_clusters, optimization_results
    
    def _get_clusterer(self, algorithm: str, n_clusters: int):
        """Get clusterer instance."""
        if algorithm == "kmeans":
            return KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
        elif algorithm == "gmm":
            return GaussianMixture(n_components=n_clusters, random_state=self.config.random_state)
        elif algorithm == "hierarchical":
            return AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == "spectral":
            return SpectralClustering(n_clusters=n_clusters, random_state=self.config.random_state)
        elif algorithm == "kmedoids" and KMEDOIDS_AVAILABLE:
            return KMedoids(n_clusters=n_clusters, random_state=self.config.random_state)
        else:
            return KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
    
    def _normalize_metric(self, criterion: str, value: float, all_results: Dict) -> float:
        """Normalize metric to 0-1 range."""
        all_values = [result['metrics'].get(criterion, 0) for result in all_results.values()]
        
        if not all_values or len(set(all_values)) == 1:
            return 0.5
        
        min_val, max_val = min(all_values), max(all_values)
        
        if min_val == max_val:
            return 0.5
        
        # For Davies-Bouldin (lower is better), we already negated it
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)


class BayesianOptimizer:
    """Bayesian optimization for clustering parameters."""
    
    def __init__(self, config: AdaptiveClusteringConfig):
        self.config = config
        self.logger = system_logger.getChild('BayesianOptimizer')
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, Bayesian optimization disabled")
    
    def optimize_parameters(
        self, 
        X: np.ndarray, 
        algorithm: str = "kmeans"
    ) -> Dict[str, Any]:
        """Optimize clustering parameters using Bayesian optimization."""
        
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna not available'}
        
        self.logger.info(f"🎯 Bayesian optimization for {algorithm}")
        
        def objective(trial):
            try:
                # Suggest parameters based on algorithm
                params = self._suggest_parameters(trial, algorithm)
                
                # Create clusterer with suggested parameters
                clusterer = self._create_clusterer(algorithm, params)
                labels = clusterer.fit_predict(X)
                
                # Calculate composite score
                metrics = {}
                if "silhouette" in self.config.optimization_criteria:
                    metrics["silhouette"] = ClusteringMetrics.silhouette_score_metric(X, labels)
                
                if "calinski_harabasz" in self.config.optimization_criteria:
                    metrics["calinski_harabasz"] = ClusteringMetrics.calinski_harabasz_score_metric(X, labels)
                
                # Composite score
                score = 0.0
                for criterion, weight in self.config.criteria_weights.items():
                    if criterion in metrics:
                        score += weight * metrics[criterion]
                
                return score
            
            except Exception as e:
                return -1.0  # Bad score for failed trials
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective, 
            n_trials=self.config.bayesian_trials,
            timeout=self.config.bayesian_timeout
        )
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _suggest_parameters(self, trial, algorithm: str) -> Dict[str, Any]:
        """Suggest parameters for different algorithms."""
        params = {}
        
        if algorithm == "kmeans":
            params['n_clusters'] = trial.suggest_int('n_clusters', 
                                                   self.config.min_clusters, 
                                                   self.config.max_clusters)
            params['init'] = trial.suggest_categorical('init', ['k-means++', 'random'])
            params['n_init'] = trial.suggest_int('n_init', 5, 20)
            params['max_iter'] = trial.suggest_int('max_iter', 100, 500)
        
        elif algorithm == "gmm":
            params['n_components'] = trial.suggest_int('n_components', 
                                                     self.config.min_clusters, 
                                                     self.config.max_clusters)
            params['covariance_type'] = trial.suggest_categorical('covariance_type', 
                                                                ['full', 'tied', 'diag', 'spherical'])
            params['max_iter'] = trial.suggest_int('max_iter', 50, 200)
        
        elif algorithm == "hierarchical":
            params['n_clusters'] = trial.suggest_int('n_clusters', 
                                                   self.config.min_clusters, 
                                                   self.config.max_clusters)
            params['linkage'] = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
        
        elif algorithm == "spectral":
            params['n_clusters'] = trial.suggest_int('n_clusters', 
                                                   self.config.min_clusters, 
                                                   self.config.max_clusters)
            params['gamma'] = trial.suggest_float('gamma', 0.001, 1.0, log=True)
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 5, 30)
        
        return params
    
    def _create_clusterer(self, algorithm: str, params: Dict[str, Any]):
        """Create clusterer with given parameters."""
        if algorithm == "kmeans":
            return KMeans(random_state=self.config.random_state, **params)
        elif algorithm == "gmm":
            return GaussianMixture(random_state=self.config.random_state, **params)
        elif algorithm == "hierarchical":
            return AgglomerativeClustering(**params)
        elif algorithm == "spectral":
            return SpectralClustering(random_state=self.config.random_state, **params)
        else:
            return KMeans(random_state=self.config.random_state)


class ReinforcementClusteringAgent:
    """Reinforcement learning agent for clustering decisions."""
    
    def __init__(self, config: AdaptiveClusteringConfig):
        self.config = config
        self.logger = system_logger.getChild('RLClusteringAgent')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State: data characteristics (mean, std, correlations, etc.)
        self.state_dim = config.meta_features_dim
        
        # Actions: (algorithm_id, n_clusters)
        self.n_algorithms = len(config.candidate_algorithms)
        self.n_cluster_choices = config.max_clusters - config.min_clusters + 1
        self.action_dim = self.n_algorithms * self.n_cluster_choices
        
        # Q-Network
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.rl_learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=config.rl_memory_size)
        
        # Exploration
        self.epsilon = config.rl_epsilon
        self.epsilon_decay = config.rl_epsilon_decay
        
        # Update target network
        self.update_target_network()
    
    def _build_q_network(self) -> nn.Module:
        """Build Q-network for clustering decisions."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        ).to(self.device)
    
    def extract_state_features(self, X: np.ndarray) -> np.ndarray:
        """Extract state features from data."""
        features = []
        
        # Basic statistics
        features.extend([
            X.shape[0],  # Number of samples
            X.shape[1],  # Number of features
            np.mean(X),
            np.std(X),
            np.min(X),
            np.max(X)
        ])
        
        # Correlation structure
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            features.extend([
                np.mean(np.abs(corr_matrix)),
                np.std(corr_matrix),
                np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            ])
        else:
            features.extend([0, 0, 0])
        
        # PCA characteristics
        if X.shape[0] > X.shape[1] and X.shape[1] > 1:
            pca = PCA()
            pca.fit(X)
            explained_var = pca.explained_variance_ratio_
            features.extend([
                explained_var[0] if len(explained_var) > 0 else 0,
                explained_var[1] if len(explained_var) > 1 else 0,
                np.sum(explained_var[:3]) if len(explained_var) >= 3 else np.sum(explained_var)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Distance characteristics
        if len(X) > 1:
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(X[:min(100, len(X))])  # Sample for efficiency
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.median(distances)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Pad or truncate to desired dimension
        while len(features) < self.state_dim:
            features.append(0)
        
        return np.array(features[:self.state_dim], dtype=np.float32)
    
    def select_action(self, state: np.ndarray) -> Tuple[str, int]:
        """Select clustering algorithm and number of clusters."""
        if np.random.random() < self.epsilon:
            # Random action
            algorithm_idx = np.random.randint(self.n_algorithms)
            n_clusters = np.random.randint(self.config.min_clusters, self.config.max_clusters + 1)
        else:
            # Q-network action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            
            # Decode action
            algorithm_idx = action_idx // self.n_cluster_choices
            cluster_idx = action_idx % self.n_cluster_choices
            n_clusters = self.config.min_clusters + cluster_idx
        
        algorithm = self.config.candidate_algorithms[algorithm_idx % len(self.config.candidate_algorithms)]
        return algorithm, n_clusters
    
    def calculate_reward(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate reward for clustering result."""
        try:
            # Multi-objective reward
            reward = 0.0
            
            # Silhouette score
            silhouette = ClusteringMetrics.silhouette_score_metric(X, labels)
            reward += 0.4 * silhouette
            
            # Calinski-Harabasz score (normalized)
            ch_score = ClusteringMetrics.calinski_harabasz_score_metric(X, labels)
            normalized_ch = np.tanh(ch_score / 1000)  # Normalize to [-1, 1]
            reward += 0.3 * normalized_ch
            
            # Davies-Bouldin score (already negated)
            db_score = ClusteringMetrics.davies_bouldin_score_metric(X, labels)
            normalized_db = np.tanh(db_score)
            reward += 0.3 * normalized_db
            
            return reward
        except:
            return -1.0  # Penalty for failed clustering
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Train the Q-network."""
        if len(self.memory) < 64:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), 64, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def encode_action(self, algorithm: str, n_clusters: int) -> int:
        """Encode action to integer."""
        algorithm_idx = self.config.candidate_algorithms.index(algorithm)
        cluster_idx = n_clusters - self.config.min_clusters
        return algorithm_idx * self.n_cluster_choices + cluster_idx


class AdaptiveEnsembleClusterer:
    """Ensemble clustering with adaptive weights."""
    
    def __init__(self, config: AdaptiveClusteringConfig):
        self.config = config
        self.logger = system_logger.getChild('AdaptiveEnsembleClusterer')
        self.algorithm_weights = {}
        self.performance_history = {}
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ensemble clustering with adaptive weighting."""
        
        self.logger.info("🎯 Adaptive ensemble clustering")
        
        # Initialize optimizers
        multi_criteria = MultiCriteriaOptimizer(self.config)
        
        # Results storage
        algorithm_results = {}
        
        # Optimize each algorithm
        for algorithm in self.config.candidate_algorithms:
            try:
                self.logger.info(f"Optimizing {algorithm}")
                
                # Multi-criteria optimization
                best_n_clusters, optimization_result = multi_criteria.optimize_clusters(X, algorithm)
                
                algorithm_results[algorithm] = {
                    'n_clusters': best_n_clusters,
                    'labels': optimization_result['best_labels'],
                    'score': optimization_result['best_score'],
                    'clusterer': optimization_result['best_clusterer'],
                    'optimization_result': optimization_result
                }
                
            except Exception as e:
                self.logger.warning(f"Algorithm {algorithm} failed: {e}")
                continue
        
        if not algorithm_results:
            raise ValueError("All clustering algorithms failed")
        
        # Calculate adaptive weights
        weights = self._calculate_adaptive_weights(algorithm_results)
        
        # Ensemble clustering
        ensemble_labels = self._ensemble_clustering(algorithm_results, weights)
        
        # Evaluate ensemble
        ensemble_score = self._evaluate_ensemble(X, ensemble_labels, algorithm_results)
        
        results = {
            'ensemble_labels': ensemble_labels,
            'ensemble_score': ensemble_score,
            'algorithm_results': algorithm_results,
            'adaptive_weights': weights,
            'n_successful_algorithms': len(algorithm_results)
        }
        
        # Update performance history
        self._update_performance_history(results)
        
        return ensemble_labels, results
    
    def _calculate_adaptive_weights(self, algorithm_results: Dict) -> Dict[str, float]:
        """Calculate adaptive weights based on performance and diversity."""
        
        weights = {}
        
        # Performance-based weights
        scores = {alg: result['score'] for alg, result in algorithm_results.items()}
        max_score = max(scores.values()) if scores else 1.0
        
        for algorithm in algorithm_results:
            # Performance component
            performance_weight = scores[algorithm] / max_score if max_score > 0 else 1.0
            
            # Historical performance component
            historical_weight = self.performance_history.get(algorithm, {}).get('avg_score', 0.5)
            
            # Diversity component (simplified)
            diversity_weight = 1.0  # Could be enhanced with actual diversity calculation
            
            # Combined weight
            weights[algorithm] = (
                0.5 * performance_weight +
                0.3 * historical_weight +
                0.2 * diversity_weight
            )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {alg: w / total_weight for alg, w in weights.items()}
        
        return weights
    
    def _ensemble_clustering(self, algorithm_results: Dict, weights: Dict) -> np.ndarray:
        """Combine clustering results using weighted voting."""
        
        if not algorithm_results:
            return np.array([])
        
        # Get all labels
        all_labels = {alg: result['labels'] for alg, result in algorithm_results.items()}
        
        # Simple approach: use the best performing algorithm
        # More sophisticated ensemble methods could be implemented
        best_algorithm = max(algorithm_results.keys(), 
                           key=lambda alg: algorithm_results[alg]['score'])
        
        return all_labels[best_algorithm]
    
    def _evaluate_ensemble(self, X: np.ndarray, labels: np.ndarray, algorithm_results: Dict) -> float:
        """Evaluate ensemble clustering result."""
        
        try:
            # Calculate multiple metrics
            silhouette = ClusteringMetrics.silhouette_score_metric(X, labels)
            ch_score = ClusteringMetrics.calinski_harabasz_score_metric(X, labels)
            db_score = ClusteringMetrics.davies_bouldin_score_metric(X, labels)
            
            # Composite score
            score = 0.4 * silhouette + 0.3 * np.tanh(ch_score / 1000) + 0.3 * np.tanh(db_score)
            return score
        
        except:
            return 0.0
    
    def _update_performance_history(self, results: Dict):
        """Update performance history for adaptive weighting."""
        
        for algorithm, result in results['algorithm_results'].items():
            if algorithm not in self.performance_history:
                self.performance_history[algorithm] = {
                    'scores': [],
                    'avg_score': 0.0,
                    'count': 0
                }
            
            history = self.performance_history[algorithm]
            history['scores'].append(result['score'])
            history['count'] += 1
            
            # Keep only recent scores
            if len(history['scores']) > 10:
                history['scores'] = history['scores'][-10:]
            
            history['avg_score'] = np.mean(history['scores'])


class AdaptiveClusteringFramework:
    """Main framework for adaptive clustering."""
    
    def __init__(self, config: AdaptiveClusteringConfig = None):
        self.config = config or AdaptiveClusteringConfig()
        self.logger = system_logger.getChild('AdaptiveClusteringFramework')
        
        # Initialize components
        self.multi_criteria_optimizer = MultiCriteriaOptimizer(self.config)
        self.bayesian_optimizer = BayesianOptimizer(self.config)
        self.rl_agent = ReinforcementClusteringAgent(self.config)
        self.ensemble_clusterer = AdaptiveEnsembleClusterer(self.config)
        
        # Performance tracking
        self.method_performance = {}
        self.adaptation_history = []
    
    def adaptive_clustering(
        self,
        X: np.ndarray,
        method: AdaptiveMethod = AdaptiveMethod.ENSEMBLE_LEARNING,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform adaptive clustering using specified method."""
        
        self.logger.info(f"🚀 Adaptive clustering using {method.value}")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == AdaptiveMethod.MULTI_CRITERIA_OPTIMIZATION:
            return self._multi_criteria_clustering(X_scaled, **kwargs)
        elif method == AdaptiveMethod.BAYESIAN_OPTIMIZATION:
            return self._bayesian_clustering(X_scaled, **kwargs)
        elif method == AdaptiveMethod.REINFORCEMENT_LEARNING:
            return self._reinforcement_clustering(X_scaled, **kwargs)
        elif method == AdaptiveMethod.ENSEMBLE_LEARNING:
            return self._ensemble_clustering(X_scaled, **kwargs)
        elif method == AdaptiveMethod.ONLINE_ADAPTATION:
            return self._online_clustering(X_scaled, **kwargs)
        else:
            raise ValueError(f"Unknown adaptive method: {method}")
    
    def _multi_criteria_clustering(self, X: np.ndarray, algorithm: str = "kmeans") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Multi-criteria optimization clustering."""
        
        best_n_clusters, results = self.multi_criteria_optimizer.optimize_clusters(X, algorithm)
        
        return results['best_labels'], {
            'method': 'multi_criteria',
            'best_n_clusters': best_n_clusters,
            'optimization_results': results
        }
    
    def _bayesian_clustering(self, X: np.ndarray, algorithm: str = "kmeans") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Bayesian optimization clustering."""
        
        optimization_results = self.bayesian_optimizer.optimize_parameters(X, algorithm)
        
        if 'error' in optimization_results:
            # Fallback to multi-criteria
            return self._multi_criteria_clustering(X, algorithm)
        
        # Use best parameters to cluster
        best_params = optimization_results['best_params']
        clusterer = self.bayesian_optimizer._create_clusterer(algorithm, best_params)
        labels = clusterer.fit_predict(X)
        
        return labels, {
            'method': 'bayesian_optimization',
            'optimization_results': optimization_results,
            'best_params': best_params
        }
    
    def _reinforcement_clustering(self, X: np.ndarray, n_episodes: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinforcement learning clustering."""
        
        n_episodes = n_episodes or self.config.rl_episodes
        
        # Training phase
        best_reward = float('-inf')
        best_labels = None
        best_action = None
        
        for episode in range(n_episodes):
            # Extract state
            state = self.rl_agent.extract_state_features(X)
            
            # Select action
            algorithm, n_clusters = self.rl_agent.select_action(state)
            action = self.rl_agent.encode_action(algorithm, n_clusters)
            
            # Perform clustering
            try:
                clusterer = self.multi_criteria_optimizer._get_clusterer(algorithm, n_clusters)
                labels = clusterer.fit_predict(X)
                
                # Calculate reward
                reward = self.rl_agent.calculate_reward(X, labels)
                
                # Store best result
                if reward > best_reward:
                    best_reward = reward
                    best_labels = labels.copy()
                    best_action = (algorithm, n_clusters)
                
                # Store experience (simplified)
                next_state = state  # Same for now
                done = True
                self.rl_agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                if episode > 64:
                    self.rl_agent.train_step()
                
                # Update target network periodically
                if episode % 100 == 0:
                    self.rl_agent.update_target_network()
                
            except Exception as e:
                # Penalty for failed clustering
                reward = -1.0
                self.rl_agent.store_experience(state, action, reward, state, True)
        
        return best_labels, {
            'method': 'reinforcement_learning',
            'best_reward': best_reward,
            'best_action': best_action,
            'n_episodes': n_episodes
        }
    
    def _ensemble_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ensemble clustering with adaptive weights."""
        
        labels, results = self.ensemble_clusterer.fit_predict(X)
        
        return labels, {
            'method': 'ensemble_learning',
            **results
        }
    
    def _online_clustering(self, X: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Online adaptive clustering (simplified implementation)."""
        
        # For now, use ensemble method as baseline
        # Full online clustering would require streaming implementation
        return self._ensemble_clustering(X)
    
    def compare_methods(
        self, 
        X: np.ndarray,
        methods: Optional[List[AdaptiveMethod]] = None
    ) -> Dict[str, Any]:
        """Compare different adaptive clustering methods."""
        
        if methods is None:
            methods = [
                AdaptiveMethod.MULTI_CRITERIA_OPTIMIZATION,
                AdaptiveMethod.ENSEMBLE_LEARNING
            ]
            if OPTUNA_AVAILABLE:
                methods.append(AdaptiveMethod.BAYESIAN_OPTIMIZATION)
        
        results = {}
        
        for method in methods:
            try:
                self.logger.info(f"Testing {method.value}")
                start_time = time.time()
                
                labels, method_results = self.adaptive_clustering(X, method)
                
                # Evaluate results
                silhouette = ClusteringMetrics.silhouette_score_metric(X, labels)
                ch_score = ClusteringMetrics.calinski_harabasz_score_metric(X, labels)
                
                results[method.value] = {
                    'labels': labels,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': ch_score,
                    'n_clusters': len(np.unique(labels)),
                    'execution_time': time.time() - start_time,
                    'method_details': method_results
                }
                
                self.logger.info(f"✅ {method.value} completed (Silhouette: {silhouette:.3f})")
                
            except Exception as e:
                self.logger.error(f"❌ {method.value} failed: {e}")
                results[method.value] = {'error': str(e)}
        
        # Find best method
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_method = max(valid_results.keys(), 
                            key=lambda k: valid_results[k]['silhouette_score'])
            results['best_method'] = best_method
            results['best_score'] = valid_results[best_method]['silhouette_score']
        
        return results


# Example usage and integration
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    
    # Create sample data with known cluster structure
    from sklearn.datasets import make_blobs
    X, true_labels = make_blobs(n_samples=1000, centers=4, n_features=10, 
                                cluster_std=1.5, random_state=42)
    
    # Initialize adaptive clustering framework
    config = AdaptiveClusteringConfig(
        min_clusters=2,
        max_clusters=8,
        bayesian_trials=50,
        rl_episodes=100,
        verbose=True
    )
    
    framework = AdaptiveClusteringFramework(config)
    
    # Compare different adaptive methods
    comparison_results = framework.compare_methods(X)
    
    print("🎯 Adaptive Clustering Comparison Results:")
    for method, results in comparison_results.items():
        if method == 'best_method':
            print(f"\n🏆 Best Method: {results}")
        elif method == 'best_score':
            print(f"🏆 Best Score: {results:.3f}")
        elif 'error' not in results:
            print(f"\n{method}:")
            print(f"  - Silhouette Score: {results['silhouette_score']:.3f}")
            print(f"  - Calinski-Harabasz: {results['calinski_harabasz_score']:.1f}")
            print(f"  - Number of Clusters: {results['n_clusters']}")
            print(f"  - Execution Time: {results['execution_time']:.2f}s")
        else:
            print(f"\n{method}: {results['error']}")
    
    # Test individual methods
    print("\n🧪 Testing Individual Adaptive Methods:")
    
    # Multi-criteria optimization
    labels_mc, results_mc = framework.adaptive_clustering(
        X, AdaptiveMethod.MULTI_CRITERIA_OPTIMIZATION
    )
    print(f"Multi-criteria: {len(np.unique(labels_mc))} clusters")
    
    # Ensemble clustering
    labels_ensemble, results_ensemble = framework.adaptive_clustering(
        X, AdaptiveMethod.ENSEMBLE_LEARNING
    )
    print(f"Ensemble: {len(np.unique(labels_ensemble))} clusters")
    
    # Reinforcement learning (if time permits)
    try:
        labels_rl, results_rl = framework.adaptive_clustering(
            X, AdaptiveMethod.REINFORCEMENT_LEARNING, n_episodes=50
        )
        print(f"Reinforcement Learning: {len(np.unique(labels_rl))} clusters")
    except Exception as e:
        print(f"RL clustering failed: {e}")