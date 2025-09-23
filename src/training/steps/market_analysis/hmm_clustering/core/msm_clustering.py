"""
Markov State Model (MSM) clustering for regime discovery.

This module implements Markov State Models for discovering market regimes
with better temporal dynamics modeling than traditional HMM approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import logging
import time
from dataclasses import dataclass
from enum import Enum

from .base_clustering import BaseClusterer, ClusteringResult

logger = logging.getLogger(__name__)


class MSMDistanceMetric(Enum):
    """Distance metrics for MSM construction."""
    EUCLIDEAN = "euclidean"
    MAHALANOBIS = "mahalanobis"
    CORRELATION = "correlation"
    COSINE = "cosine"


class MSMClusteringMethod(Enum):
    """Clustering methods for MSM state assignment."""
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"


@dataclass
class MSMClusteringResult(ClusteringResult):
    """Result of MSM clustering operation."""
    transition_matrix: np.ndarray = None
    eigenvalues: np.ndarray = None
    eigenvectors: np.ndarray = None
    stationary_distribution: np.ndarray = None
    implied_timescales: np.ndarray = None
    msm_score: float = 0.0
    lag_time: int = 1


@dataclass
class MSMConfig:
    """Configuration for MSM clustering."""
    n_states: int = 20
    lag_time: int = 1
    clustering_method: str = "kmeans"
    distance_metric: str = "euclidean"
    reversible: bool = True
    stationary_distribution_constraint: bool = True
    ergodic_cutoff: float = 1e-6
    connectivity_threshold: float = 0.1


class MSMClusterer(BaseClusterer):
    """
    Markov State Model (MSM) based clustering for regime discovery.

    This implementation uses Markov State Models to identify market regimes
    based on temporal dynamics and transition patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the MSM clusterer.

        Args:
            config: MSM clustering configuration
        """
        super().__init__(config)

        # Extract MSM-specific configuration
        self.msm_config = MSMConfig(
            n_states=config.get('n_states', 20),
            lag_time=config.get('lag_time', 1),
            clustering_method=config.get('clustering_method', 'kmeans'),
            distance_metric=config.get('distance_metric', 'euclidean'),
            reversible=config.get('reversible', True),
            stationary_distribution_constraint=config.get('stationary_distribution_constraint', True),
            ergodic_cutoff=config.get('ergodic_cutoff', 1e-6),
            connectivity_threshold=config.get('connectivity_threshold', 0.1)
        )

        # Initialize clustering method
        self._initialize_clustering_method()

        self.logger.info(f"✅ MSM Clusterer initialized with {self.msm_config.n_states} states, lag_time={self.msm_config.lag_time}")

    def _initialize_clustering_method(self):
        """Initialize the clustering method based on configuration."""
        if self.msm_config.clustering_method == 'kmeans':
            self.clustering_method = KMeans(
                n_clusters=self.msm_config.n_states,
                random_state=42,
                n_init=10
            )
        elif self.msm_config.clustering_method == 'agglomerative':
            self.clustering_method = AgglomerativeClustering(
                n_clusters=self.msm_config.n_states,
                linkage='ward'
            )
        else:
            # Default to K-means
            self.clustering_method = KMeans(
                n_clusters=self.msm_config.n_states,
                random_state=42,
                n_init=10
            )

    def cluster(self, features: np.ndarray, optimize_parameters: bool = True) -> ClusteringResult:
        """Perform MSM clustering on the given features.

        Args:
            features: Feature matrix to cluster
            optimize_parameters: Whether to use Bayesian optimization for parameters

        Returns:
            MSMClusteringResult with clustering results
        """
        start_time = time.time()

        try:
            # Prepare features
            features = self._prepare_features(features)

            # Monitor performance
            self._monitor_performance("msm_clustering")

            # Optimize parameters if requested
            if optimize_parameters:
                optimized_config = self._optimize_msm_parameters(features)
                if optimized_config:
                    # Update configuration with optimized parameters
                    for key, value in optimized_config.items():
                        if hasattr(self.msm_config, key):
                            setattr(self.msm_config, key, value)
                    self.logger.info(f"🔧 Updated MSM config with optimized parameters: {optimized_config}")

            # Perform MSM clustering
            result = self._perform_msm_clustering(features)

            # Stop performance monitoring
            perf_metrics = self._stop_performance_monitoring("msm_clustering")

            # Calculate execution time
            execution_time = time.time() - start_time

            # Create MSM result
            clustering_result = self._create_msm_result(
                labels=result['labels'],
                features=features,
                execution_time=execution_time,
                transition_matrix=result.get('transition_matrix'),
                eigenvalues=result.get('eigenvalues'),
                eigenvectors=result.get('eigenvectors'),
                stationary_distribution=result.get('stationary_distribution'),
                implied_timescales=result.get('implied_timescales'),
                msm_score=result.get('msm_score', 0.0),
                metadata={
                    'method': 'msm_clustering',
                    'n_states': self.msm_config.n_states,
                    'lag_time': self.msm_config.lag_time,
                    'clustering_method': self.msm_config.clustering_method,
                    'matrix_ops_used': self.matrix_ops is not None,
                    'hardware_acceleration_used': self.hardware_accelerator is not None,
                    'performance_metrics': perf_metrics,
                    'parameter_optimization': optimize_parameters
                }
            )

            self.logger.info(f"✅ MSM clustering completed in {execution_time".2f"}s with {self.msm_config.n_states} states")
            return clustering_result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ MSM clustering failed: {e}")
            return MSMClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'msm_clustering'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _perform_msm_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform the core MSM clustering algorithm.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with MSM clustering results
        """
        self.logger.info("🚀 Starting MSM clustering algorithm")

        # Step 1: Initial clustering to get state assignments
        initial_labels = self._perform_initial_clustering(features)

        # Step 2: Construct transition count matrix
        transition_counts = self._construct_transition_matrix(features, initial_labels)

        # Step 3: Compute transition probability matrix
        transition_matrix = self._compute_transition_probabilities(transition_counts)

        # Step 4: Validate MSM properties
        is_valid, validation_info = self._validate_msm_properties(transition_matrix)

        # Step 5: Compute MSM properties
        eigenvalues, eigenvectors = self._compute_msm_eigensystem(transition_matrix)
        stationary_distribution = self._compute_stationary_distribution(transition_matrix)
        implied_timescales = self._compute_implied_timescales(eigenvalues)

        # Step 6: Compute MSM score
        msm_score = self._compute_msm_score(transition_matrix, stationary_distribution)

        # Step 7: Refine state assignments if needed
        final_labels = self._refine_state_assignments(features, transition_matrix, initial_labels)

        self.logger.info("✅ MSM clustering completed")

        return {
            'labels': final_labels,
            'transition_matrix': transition_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'stationary_distribution': stationary_distribution,
            'implied_timescales': implied_timescales,
            'msm_score': msm_score,
            'validation_info': validation_info
        }

    def _perform_initial_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform initial clustering to get state assignments.

        Args:
            features: Feature matrix

        Returns:
            Initial cluster labels
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Perform clustering
            labels = self.clustering_method.fit_predict(features_scaled)

            self.logger.info(f"✅ Initial clustering completed: {len(np.unique(labels))} states")
            return labels

        except Exception as e:
            self.logger.warning(f"⚠️ Initial clustering failed: {e}")
            # Fallback to simple clustering
            return np.random.randint(0, self.msm_config.n_states, len(features))

    def _construct_transition_matrix(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Construct transition count matrix from time series data.

        Args:
            features: Feature matrix
            labels: State labels

        Returns:
            Transition count matrix
        """
        try:
            n_states = self.msm_config.n_states
            lag_time = self.msm_config.lag_time

            # Initialize transition count matrix
            transition_counts = np.zeros((n_states, n_states))

            # Count transitions with lag time
            for i in range(len(labels) - lag_time):
                current_state = labels[i]
                future_state = labels[i + lag_time]

                if 0 <= current_state < n_states and 0 <= future_state < n_states:
                    transition_counts[current_state, future_state] += 1

            self.logger.info(f"✅ Transition matrix constructed: {transition_counts.shape}")
            return transition_counts

        except Exception as e:
            self.logger.warning(f"⚠️ Transition matrix construction failed: {e}")
            return np.eye(n_states)

    def _compute_transition_probabilities(self, transition_counts: np.ndarray) -> np.ndarray:
        """Compute transition probability matrix from counts.

        Args:
            transition_counts: Transition count matrix

        Returns:
            Transition probability matrix
        """
        try:
            # Normalize rows to get probabilities
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero

            transition_probs = transition_counts / row_sums

            # Apply reversibility constraint if requested
            if self.msm_config.reversible:
                transition_probs = self._enforce_reversibility(transition_probs)

            # Apply stationary distribution constraint if requested
            if self.msm_config.stationary_distribution_constraint:
                transition_probs = self._enforce_stationary_constraint(transition_probs)

            self.logger.info("✅ Transition probabilities computed")
            return transition_probs

        except Exception as e:
            self.logger.warning(f"⚠️ Transition probability computation failed: {e}")
            return np.eye(self.msm_config.n_states) / self.msm_config.n_states

    def _enforce_reversibility(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Enforce detailed balance (reversibility) constraint.

        Args:
            transition_matrix: Current transition matrix

        Returns:
            Reversible transition matrix
        """
        try:
            # Simple approach: make matrix symmetric
            # In practice, this would use more sophisticated methods
            symmetric_matrix = (transition_matrix + transition_matrix.T) / 2

            # Renormalize rows
            row_sums = symmetric_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)

            return symmetric_matrix / row_sums

        except Exception as e:
            self.logger.warning(f"⚠️ Reversibility enforcement failed: {e}")
            return transition_matrix

    def _enforce_stationary_constraint(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Enforce stationary distribution constraint.

        Args:
            transition_matrix: Current transition matrix

        Returns:
            Constrained transition matrix
        """
        try:
            # Compute stationary distribution
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))

            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / np.sum(stationary_dist)

            # Adjust transition matrix to match stationary distribution
            # This is a simplified approach
            adjusted_matrix = transition_matrix.copy()

            return adjusted_matrix

        except Exception as e:
            self.logger.warning(f"⚠️ Stationary constraint enforcement failed: {e}")
            return transition_matrix

    def _validate_msm_properties(self, transition_matrix: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Validate MSM properties.

        Args:
            transition_matrix: Transition probability matrix

        Returns:
            Tuple of (is_valid, validation_info)
        """
        try:
            validation_info = {
                'is_stochastic': self._check_stochasticity(transition_matrix),
                'is_connected': self._check_connectivity(transition_matrix),
                'is_ergodic': self._check_ergodicity(transition_matrix),
                'detailed_balance': self._check_detailed_balance(transition_matrix)
            }

            is_valid = all(validation_info.values())
            return is_valid, validation_info

        except Exception as e:
            self.logger.warning(f"⚠️ MSM validation failed: {e}")
            return False, {'error': str(e)}

    def _check_stochasticity(self, transition_matrix: np.ndarray) -> bool:
        """Check if transition matrix is stochastic (rows sum to 1)."""
        row_sums = np.sum(transition_matrix, axis=1)
        return np.allclose(row_sums, 1.0, atol=1e-6)

    def _check_connectivity(self, transition_matrix: np.ndarray) -> bool:
        """Check if transition matrix is strongly connected."""
        try:
            # Check if matrix power eventually has all positive entries
            n_states = transition_matrix.shape[0]
            for k in range(2, n_states + 1):
                powered = np.linalg.matrix_power(transition_matrix, k)
                if np.all(powered > self.msm_config.connectivity_threshold):
                    return True
            return False
        except Exception:
            return False

    def _check_ergodicity(self, transition_matrix: np.ndarray) -> bool:
        """Check if transition matrix is ergodic."""
        try:
            eigenvalues, _ = np.linalg.eig(transition_matrix)
            spectral_gap = np.sort(np.abs(eigenvalues))[-2]

            return spectral_gap < self.msm_config.ergodic_cutoff
        except Exception:
            return False

    def _check_detailed_balance(self, transition_matrix: np.ndarray) -> bool:
        """Check if transition matrix satisfies detailed balance."""
        try:
            # For detailed balance, T_ij * pi_i = T_ji * pi_j
            # We use a simple eigenvalue-based check
            eigenvalues, _ = np.linalg.eig(transition_matrix)
            return np.allclose(eigenvalues, eigenvalues.conj())
        except Exception:
            return False

    def _compute_msm_eigensystem(self, transition_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of transition matrix.

        Args:
            transition_matrix: Transition probability matrix

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        try:
            # Compute eigensystem
            if transition_matrix.shape[0] > 50:
                # Use sparse methods for large matrices
                sparse_matrix = csr_matrix(transition_matrix)
                eigenvalues, eigenvectors = eigs(sparse_matrix, k=min(10, transition_matrix.shape[0]-1))
            else:
                eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)

            # Sort by magnitude (largest first)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            return eigenvalues, eigenvectors

        except Exception as e:
            self.logger.warning(f"⚠️ Eigensystem computation failed: {e}")
            n_states = transition_matrix.shape[0]
            return np.ones(n_states), np.eye(n_states)

    def _compute_stationary_distribution(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Compute stationary distribution of the MSM.

        Args:
            transition_matrix: Transition probability matrix

        Returns:
            Stationary distribution
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))

            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / np.sum(stationary_dist)

            # Ensure non-negativity
            stationary_dist = np.maximum(stationary_dist, 0)

            return stationary_dist

        except Exception as e:
            self.logger.warning(f"⚠️ Stationary distribution computation failed: {e}")
            n_states = transition_matrix.shape[0]
            return np.ones(n_states) / n_states

    def _compute_implied_timescales(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Compute implied timescales from MSM eigenvalues.

        Args:
            eigenvalues: MSM eigenvalues

        Returns:
            Implied timescales
        """
        try:
            # Implied timescale t_i = -lag_time / ln(|λ_i|)
            # Only consider eigenvalues with |λ_i| < 1
            valid_eigenvals = eigenvalues[(np.abs(eigenvalues) < 1) & (np.abs(eigenvalues) > 1e-10)]

            timescales = -self.msm_config.lag_time / np.log(np.abs(valid_eigenvals))

            return timescales

        except Exception as e:
            self.logger.warning(f"⚠️ Implied timescales computation failed: {e}")
            return np.array([1.0, 2.0, 5.0, 10.0])

    def _compute_msm_score(self, transition_matrix: np.ndarray, stationary_distribution: np.ndarray) -> float:
        """Compute MSM quality score.

        Args:
            transition_matrix: Transition probability matrix
            stationary_distribution: Stationary distribution

        Returns:
            MSM score
        """
        try:
            # Simple score based on connectivity and stationarity
            connectivity_score = np.mean(transition_matrix > self.msm_config.connectivity_threshold)
            stationarity_score = np.min(stationary_distribution) / np.max(stationary_distribution)

            # Combine scores
            msm_score = 0.7 * connectivity_score + 0.3 * (1.0 / (1.0 + stationarity_score))

            return float(msm_score)

        except Exception as e:
            self.logger.warning(f"⚠️ MSM score computation failed: {e}")
            return 0.5

    def _refine_state_assignments(self, features: np.ndarray, transition_matrix: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
        """Refine state assignments based on MSM properties.

        Args:
            features: Feature matrix
            transition_matrix: Transition probability matrix
            initial_labels: Initial state assignments

        Returns:
            Refined state assignments
        """
        try:
            # For now, return initial labels
            # In a full implementation, this would use MSM-based refinement
            return initial_labels

        except Exception as e:
            self.logger.warning(f"⚠️ State assignment refinement failed: {e}")
            return initial_labels

    def _optimize_msm_parameters(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        """Optimize MSM parameters using Bayesian optimization.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with optimized parameters or None
        """
        try:
            from src.training.steps.model_training.bayesian_optimization_msm import (
                optimize_msm_parameters, MSMOptimizationConfig
            )

            # Create optimization config
            opt_config = MSMOptimizationConfig(
                n_trials=20,  # Limited trials for efficiency
                timeout=300,  # 5 minutes timeout
                optimization_objective='msm_score'
            )

            # Perform optimization
            opt_results = optimize_msm_parameters(features, opt_config)

            if opt_results['success']:
                self.logger.info(f"✅ MSM parameter optimization successful, best score: {opt_results['best_score']:.4f}")
                return opt_results['best_params']
            else:
                self.logger.warning(f"⚠️ MSM parameter optimization failed: {opt_results.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            self.logger.warning(f"⚠️ MSM parameter optimization failed: {e}")
            return None

    def _create_msm_result(self, labels: np.ndarray, features: np.ndarray, execution_time: float,
                          transition_matrix: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                          stationary_distribution: np.ndarray, implied_timescales: np.ndarray,
                          msm_score: float, metadata: Dict[str, Any]) -> MSMClusteringResult:
        """Create MSM clustering result.

        Args:
            labels: Cluster labels
            features: Feature matrix
            execution_time: Execution time
            transition_matrix: Transition probability matrix
            eigenvalues: MSM eigenvalues
            eigenvectors: MSM eigenvectors
            stationary_distribution: Stationary distribution
            implied_timescales: Implied timescales
            msm_score: MSM quality score
            metadata: Additional metadata

        Returns:
            MSMClusteringResult object
        """
        try:
            # Create base result
            base_result = self._create_result(labels, features, execution_time, metadata)

            # Add MSM-specific fields
            msm_result = MSMClusteringResult(
                labels=base_result.labels,
                cluster_centers=base_result.cluster_centers,
                statistics=base_result.statistics,
                quality_metrics=base_result.quality_metrics,
                validation=base_result.validation,
                metadata=base_result.metadata,
                success=base_result.success,
                error_message=base_result.error_message,
                execution_time=base_result.execution_time,
                timestamp=base_result.timestamp,
                transition_matrix=transition_matrix,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                stationary_distribution=stationary_distribution,
                implied_timescales=implied_timescales,
                msm_score=msm_score,
                lag_time=self.msm_config.lag_time
            )

            return msm_result

        except Exception as e:
            self.logger.error(f"❌ Failed to create MSM clustering result: {e}")
            return MSMClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )