"""
HMM Temporal Layer for Regime Clustering

This module provides HMM-based temporal refinement for regime clustering results.
It can be used as a drop-in replacement for iterative optimization temporal stabilization.

Author: Ares Team
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


@dataclass
class HMMTemporalResult:
    """Result from HMM temporal refinement."""
    refined_labels: np.ndarray
    regime_probabilities: np.ndarray
    transition_matrix: np.ndarray
    regime_stability: Dict[int, float]
    log_likelihood: float
    convergence_info: Dict[str, Any]
    hmm_model: Any
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class HMMTemporalLayer:
    """
    HMM-based temporal refinement for regime clustering.
    
    This class takes initial regime labels (e.g., from HDBSCAN) and refines them
    using Hidden Markov Model to capture temporal dynamics and transition probabilities.
    
    Features:
    - Initialization from clustering results
    - Temporal smoothing via Viterbi algorithm
    - Transition probability estimation
    - Regime stability analysis
    - Support for multiple covariance types
    - Convergence monitoring
    
    Example:
        >>> hmm_layer = HMMTemporalLayer(n_components=5, covariance_type="full")
        >>> hmm_layer.initialize_from_clusters(features, cluster_labels)
        >>> hmm_layer.fit(features)
        >>> refined_labels = hmm_layer.predict(features)
    """
    
    def __init__(self, 
                 n_components: int,
                 covariance_type: str = "full",
                 n_iter: int = 100,
                 random_state: int = 42,
                 convergence_threshold: float = 1e-4,
                 verbose: bool = True):
        """
        Initialize HMM temporal layer.
        
        Args:
            n_components: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ("full", "diag", "spherical")
            n_iter: Maximum number of EM iterations
            random_state: Random state for reproducibility
            convergence_threshold: Convergence threshold for EM algorithm
            verbose: Whether to print progress information
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM temporal layer. "
                "Install with: pip install hmmlearn"
            )
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize HMM model
        self.hmm = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=convergence_threshold,
            verbose=0  # We'll handle verbosity
        )
        
        self.is_initialized = False
        self.is_fitted = False
        
        if self.verbose:
            tprint_info(f"HMM Temporal Layer initialized: {n_components} states, {covariance_type} covariance")
    
    def initialize_from_clusters(self, 
                                 features: np.ndarray,
                                 cluster_labels: np.ndarray) -> 'HMMTemporalLayer':
        """
        Initialize HMM parameters from clustering results.
        
        This method sets initial values for:
        - Start probabilities (from cluster frequencies)
        - Transition matrix (from observed transitions)
        - Emission means (from cluster centroids)
        - Emission covariances (from cluster covariances)
        
        Args:
            features: Feature matrix (N, n_features)
            cluster_labels: Initial cluster labels from clustering algorithm
            
        Returns:
            self for chaining
        """
        if self.verbose:
            tprint_info("Initializing HMM from cluster labels...")
        
        try:
            # Validate inputs
            if len(features) != len(cluster_labels):
                raise ValueError(f"Features and labels length mismatch: {len(features)} vs {len(cluster_labels)}")
            
            # Handle noise label (-1) if present
            valid_mask = cluster_labels >= 0
            if not valid_mask.all():
                tprint_warning(f"Found {(~valid_mask).sum()} noise points (label -1), will map to nearest cluster")
                cluster_labels = self._handle_noise_labels(features, cluster_labels)
            
            # Ensure we have the right number of clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) != self.n_components:
                tprint_warning(
                    f"Number of unique labels ({len(unique_labels)}) != n_components ({self.n_components}). "
                    "Adjusting labels..."
                )
                cluster_labels = self._adjust_label_count(cluster_labels, self.n_components)
            
            # 1. Compute initial state probabilities
            self.hmm.startprob_ = self._compute_start_probabilities(cluster_labels)
            
            # 2. Compute transition matrix from observed transitions
            self.hmm.transmat_ = self._estimate_transition_matrix(cluster_labels)
            
            # 3. Compute emission means (cluster centroids)
            self.hmm.means_ = self._compute_emission_means(features, cluster_labels)
            
            # 4. Compute emission covariances
            self.hmm.covars_ = self._compute_emission_covariances(features, cluster_labels)
            
            self.is_initialized = True
            
            if self.verbose:
                tprint_success("HMM initialized from clusters successfully")
                self._log_initialization_info()
            
            return self
            
        except Exception as e:
            tprint_error(f"Failed to initialize HMM from clusters: {e}")
            raise
    
    def _handle_noise_labels(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Map noise points to nearest valid cluster."""
        noise_mask = labels == -1
        valid_mask = ~noise_mask
        
        # Get centroids of valid clusters
        unique_valid = np.unique(labels[valid_mask])
        centroids = np.array([
            features[labels == k].mean(axis=0)
            for k in unique_valid
        ])
        
        # Assign noise points to nearest centroid
        for noise_idx in np.where(noise_mask)[0]:
            distances = np.linalg.norm(centroids - features[noise_idx], axis=1)
            nearest_cluster = unique_valid[np.argmin(distances)]
            labels[noise_idx] = nearest_cluster
        
        return labels
    
    def _adjust_label_count(self, labels: np.ndarray, target_n: int) -> np.ndarray:
        """Adjust number of unique labels to match target."""
        unique_labels = np.unique(labels)
        current_n = len(unique_labels)
        
        if current_n < target_n:
            # Need to split clusters
            # Split largest cluster(s)
            tprint_warning(f"Splitting clusters: {current_n} → {target_n}")
            # For now, just map to 0..target_n-1
            return np.mod(labels, target_n)
        else:
            # Need to merge clusters
            # Merge smallest clusters
            tprint_warning(f"Merging clusters: {current_n} → {target_n}")
            # Simple approach: map to 0..target_n-1
            return np.mod(labels, target_n)
    
    def _compute_start_probabilities(self, labels: np.ndarray) -> np.ndarray:
        """Compute initial state probabilities from label distribution."""
        counts = np.bincount(labels, minlength=self.n_components)
        probs = counts / counts.sum()
        
        # Add small probability to ensure no zero probabilities
        probs += 1e-10
        probs /= probs.sum()
        
        return probs
    
    def _estimate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from observed label transitions."""
        trans_matrix = np.zeros((self.n_components, self.n_components))
        
        # Count transitions
        for i in range(len(labels) - 1):
            trans_matrix[labels[i], labels[i+1]] += 1
        
        # Normalize rows
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(
            trans_matrix, 
            row_sums, 
            where=row_sums > 0,
            out=np.zeros_like(trans_matrix)
        )
        
        # Add small probability for unobserved transitions (smoothing)
        trans_matrix += 0.01
        trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)
        
        return trans_matrix
    
    def _compute_emission_means(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute emission means (cluster centroids)."""
        means = np.array([
            features[labels == k].mean(axis=0)
            for k in range(self.n_components)
        ])
        return means
    
    def _compute_emission_covariances(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute emission covariances."""
        if self.covariance_type == "full":
            covars = np.array([
                np.cov(features[labels == k].T) + 1e-6 * np.eye(features.shape[1])
                for k in range(self.n_components)
            ])
        elif self.covariance_type == "diag":
            covars = np.array([
                np.var(features[labels == k], axis=0) + 1e-6
                for k in range(self.n_components)
            ])
        elif self.covariance_type == "spherical":
            covars = np.array([
                np.var(features[labels == k]) + 1e-6
                for k in range(self.n_components)
            ])
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")
        
        return covars
    
    def _log_initialization_info(self):
        """Log initialization information."""
        tprint_info(f"Start probabilities: {self.hmm.startprob_}")
        tprint_info("Transition matrix diagonal (self-transitions):")
        for i in range(self.n_components):
            tprint_info(f"  Regime {i}: {self.hmm.transmat_[i, i]:.3f}")
    
    def fit(self, features: np.ndarray, lengths: Optional[np.ndarray] = None) -> 'HMMTemporalLayer':
        """
        Fit HMM to refine temporal dynamics using Baum-Welch algorithm.
        
        Args:
            features: Feature matrix (N, n_features)
            lengths: Optional array of sequence lengths for multiple sequences
            
        Returns:
            self for chaining
        """
        if not self.is_initialized:
            raise ValueError("HMM must be initialized before fitting. Call initialize_from_clusters() first.")
        
        if self.verbose:
            tprint_info("Fitting HMM to learn temporal dynamics...")
        
        try:
            start_time = datetime.now()
            
            # Fit HMM
            self.hmm.fit(features, lengths=lengths)
            
            fit_time = (datetime.now() - start_time).total_seconds()
            self.is_fitted = True
            
            if self.verbose:
                tprint_success(f"HMM fitted successfully in {fit_time:.2f} seconds")
                tprint_info(f"Converged: {self.hmm.monitor_.converged}")
                tprint_info(f"Iterations: {self.hmm.monitor_.iter}")
                tprint_info(f"Final log-likelihood: {self.hmm.monitor_.history[-1]:.2f}")
            
            return self
            
        except Exception as e:
            tprint_error(f"Failed to fit HMM: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict optimal regime sequence using Viterbi algorithm.
        
        This finds the most likely state sequence given the observations.
        
        Args:
            features: Feature matrix (N, n_features)
            
        Returns:
            Array of regime labels
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction. Call fit() first.")
        
        return self.hmm.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for each time step.
        
        Args:
            features: Feature matrix (N, n_features)
            
        Returns:
            Array of shape (N, n_components) with regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction. Call fit() first.")
        
        return self.hmm.predict_proba(features)
    
    def score(self, features: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        
        Args:
            features: Feature matrix (N, n_features)
            
        Returns:
            Log-likelihood score
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before scoring. Call fit() first.")
        
        return self.hmm.score(features)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get learned transition matrix."""
        if not self.is_fitted:
            raise ValueError("HMM must be fitted first. Call fit() first.")
        
        return self.hmm.transmat_
    
    def compute_regime_stability(self) -> Dict[int, float]:
        """
        Compute expected duration in each regime.
        
        Expected duration = 1 / (1 - P(stay in regime))
        
        Returns:
            Dictionary mapping regime ID to expected duration
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted first. Call fit() first.")
        
        stability = {}
        for i in range(self.n_components):
            p_stay = self.hmm.transmat_[i, i]
            if p_stay < 1.0:
                stability[i] = 1.0 / (1.0 - p_stay)
            else:
                stability[i] = float('inf')
        
        return stability
    
    def analyze_transitions(self) -> Dict[str, Any]:
        """
        Analyze transition patterns.
        
        Returns:
            Dictionary with transition analysis
        """
        trans_matrix = self.get_transition_matrix()
        stability = self.compute_regime_stability()
        
        # Find most likely transitions
        most_likely_transitions = []
        for i in range(self.n_components):
            # Get transitions FROM regime i
            trans_probs = trans_matrix[i]
            # Sort by probability (excluding self-transition)
            trans_probs_no_self = trans_probs.copy()
            trans_probs_no_self[i] = 0
            
            if trans_probs_no_self.sum() > 0:
                most_likely_to = np.argmax(trans_probs_no_self)
                most_likely_transitions.append({
                    'from': i,
                    'to': most_likely_to,
                    'probability': trans_probs[most_likely_to]
                })
        
        return {
            'transition_matrix': trans_matrix,
            'regime_stability': stability,
            'most_likely_transitions': most_likely_transitions,
            'average_stability': np.mean(list(stability.values()))
        }
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information from fitting."""
        if not self.is_fitted:
            return {}
        
        return {
            'converged': self.hmm.monitor_.converged,
            'iterations': self.hmm.monitor_.iter,
            'history': list(self.hmm.monitor_.history),
            'final_log_likelihood': self.hmm.monitor_.history[-1] if self.hmm.monitor_.history else None
        }


async def refine_with_hmm(hdbscan_result: Any,
                         features_df: pd.DataFrame,
                         config: Dict[str, Any]) -> HMMTemporalResult:
    """
    Refine HDBSCAN results with HMM temporal modeling.
    
    This is the main integration function that takes HDBSCAN clustering results
    and refines them using HMM to capture temporal dynamics.
    
    Args:
        hdbscan_result: Result object from HDBSCAN regime discovery
        features_df: DataFrame with features used for clustering
        config: Configuration dictionary with HMM settings
        
    Returns:
        HMMTemporalResult with refined labels and analysis
    """
    try:
        tprint_info("🔄 Starting HMM temporal refinement...")
        
        # Extract relevant data
        if hasattr(hdbscan_result, 'cluster_labels'):
            initial_labels = hdbscan_result.cluster_labels
        elif hasattr(hdbscan_result, 'labels'):
            initial_labels = hdbscan_result.labels
        else:
            raise ValueError("Could not find cluster labels in hdbscan_result")
        
        # Count non-noise regimes
        unique_labels = np.unique(initial_labels[initial_labels != -1])
        n_regimes = len(unique_labels)
        
        if n_regimes < 2:
            tprint_warning("Less than 2 regimes found, skipping HMM refinement")
            return HMMTemporalResult(
                refined_labels=initial_labels,
                regime_probabilities=np.zeros((len(initial_labels), n_regimes)),
                transition_matrix=np.eye(max(2, n_regimes)),
                regime_stability={},
                log_likelihood=0.0,
                convergence_info={},
                hmm_model=None,
                metadata={'skipped': True, 'reason': 'insufficient_regimes'},
                success=True
            )
        
        tprint_info(f"Refining {n_regimes} regimes with HMM")
        
        # Get HMM configuration
        hmm_config = config.get('hmm_config', {})
        covariance_type = hmm_config.get('covariance_type', 'full')
        n_iter = hmm_config.get('n_iter', 100)
        convergence_threshold = hmm_config.get('convergence_threshold', 1e-4)
        
        # Create HMM temporal layer
        hmm_layer = HMMTemporalLayer(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            convergence_threshold=convergence_threshold,
            verbose=True
        )
        
        # Initialize from HDBSCAN results
        hmm_layer.initialize_from_clusters(features_df.values, initial_labels)
        
        # Fit HMM to learn temporal dynamics
        hmm_layer.fit(features_df.values)
        
        # Predict refined regime sequence
        refined_labels = hmm_layer.predict(features_df.values)
        regime_probs = hmm_layer.predict_proba(features_df.values)
        log_likelihood = hmm_layer.score(features_df.values)
        
        # Get transition analysis
        transition_analysis = hmm_layer.analyze_transitions()
        convergence_info = hmm_layer.get_convergence_info()
        
        # Calculate improvement metrics
        temporal_coherence_before = _compute_temporal_coherence(initial_labels)
        temporal_coherence_after = _compute_temporal_coherence(refined_labels)
        improvement = temporal_coherence_after - temporal_coherence_before
        
        tprint_success(f"✅ HMM refinement complete!")
        tprint_info(f"Temporal coherence: {temporal_coherence_before:.3f} → {temporal_coherence_after:.3f} (+{improvement:.3f})")
        tprint_info(f"Average regime stability: {transition_analysis['average_stability']:.1f} timesteps")
        
        return HMMTemporalResult(
            refined_labels=refined_labels,
            regime_probabilities=regime_probs,
            transition_matrix=transition_analysis['transition_matrix'],
            regime_stability=transition_analysis['regime_stability'],
            log_likelihood=log_likelihood,
            convergence_info=convergence_info,
            hmm_model=hmm_layer.hmm,
            metadata={
                'n_regimes': n_regimes,
                'temporal_coherence_before': temporal_coherence_before,
                'temporal_coherence_after': temporal_coherence_after,
                'improvement': improvement,
                'most_likely_transitions': transition_analysis['most_likely_transitions'],
                'covariance_type': covariance_type,
                'converged': convergence_info.get('converged', False)
            },
            success=True
        )
        
    except Exception as e:
        tprint_error(f"❌ HMM refinement failed: {e}")
        return HMMTemporalResult(
            refined_labels=np.array([]),
            regime_probabilities=np.array([]),
            transition_matrix=np.array([]),
            regime_stability={},
            log_likelihood=0.0,
            convergence_info={},
            hmm_model=None,
            metadata={},
            success=False,
            error_message=str(e)
        )


def _compute_temporal_coherence(labels: np.ndarray) -> float:
    """
    Compute temporal coherence metric.
    
    Measures how stable regime assignments are over time.
    Higher values indicate more temporally coherent regimes.
    
    Args:
        labels: Regime labels
        
    Returns:
        Temporal coherence score (0-1)
    """
    if len(labels) < 2:
        return 0.0
    
    # Count regime transitions
    transitions = np.sum(labels[1:] != labels[:-1])
    
    # Normalize by maximum possible transitions
    max_transitions = len(labels) - 1
    
    # Coherence = 1 - (transition_rate)
    coherence = 1.0 - (transitions / max_transitions)
    
    return coherence


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_hmm_temporal_layer():
        """Test HMM temporal layer with synthetic data."""
        print("Testing HMM Temporal Layer\n" + "="*50)
        
        # Generate synthetic regime data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        n_regimes = 3
        
        # Generate features with regime structure
        features = []
        true_labels = []
        
        for regime in range(n_regimes):
            n_regime_samples = n_samples // n_regimes
            regime_mean = np.random.randn(n_features) * 5
            regime_features = np.random.randn(n_regime_samples, n_features) + regime_mean
            features.append(regime_features)
            true_labels.extend([regime] * n_regime_samples)
        
        features = np.vstack(features)
        true_labels = np.array(true_labels)
        
        # Add some noise to labels (simulating imperfect clustering)
        noisy_labels = true_labels.copy()
        noise_mask = np.random.rand(len(noisy_labels)) < 0.1  # 10% noise
        noisy_labels[noise_mask] = np.random.randint(0, n_regimes, noise_mask.sum())
        
        print(f"Generated {n_samples} samples, {n_features} features, {n_regimes} regimes")
        print(f"True temporal coherence: {_compute_temporal_coherence(true_labels):.3f}")
        print(f"Noisy temporal coherence: {_compute_temporal_coherence(noisy_labels):.3f}\n")
        
        # Test HMM temporal layer
        hmm_layer = HMMTemporalLayer(
            n_components=n_regimes,
            covariance_type="full",
            verbose=True
        )
        
        hmm_layer.initialize_from_clusters(features, noisy_labels)
        hmm_layer.fit(features)
        
        refined_labels = hmm_layer.predict(features)
        regime_probs = hmm_layer.predict_proba(features)
        
        print(f"\nRefined temporal coherence: {_compute_temporal_coherence(refined_labels):.3f}")
        
        # Transition analysis
        transition_analysis = hmm_layer.analyze_transitions()
        print("\nTransition Matrix:")
        print(transition_analysis['transition_matrix'])
        print("\nRegime Stability (expected duration):")
        for regime, duration in transition_analysis['regime_stability'].items():
            print(f"  Regime {regime}: {duration:.1f} timesteps")
        
        print("\nMost Likely Transitions:")
        for trans in transition_analysis['most_likely_transitions']:
            print(f"  Regime {trans['from']} → Regime {trans['to']}: {trans['probability']:.3f}")
    
    # Run test
    asyncio.run(test_hmm_temporal_layer())
