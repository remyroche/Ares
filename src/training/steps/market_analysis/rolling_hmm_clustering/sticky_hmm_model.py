"""
Sticky HMM Model Implementation

This module implements a Sticky HMM using hmmlearn with:
- Sticky priors for self-transition regularization
- Diagonal covariance with regularization (min_covar)
- KMeans++ initialization
- Covariance floor to avoid singular matrices
- Optimized for Mac M1

Key Features:
- Pre-fit initialization of transition matrix with sticky priors
- Post-fit regularization of learned transition matrix
- Diagonal covariance type (safer than full)
- Eigenvalue regularization for numerical stability
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings

from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error

logger = logging.getLogger(__name__)

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


@dataclass
class StickyHMMConfig:
    """Configuration for Sticky HMM model."""
    n_components: int = 5               # Number of hidden states (4-6 recommended)
    n_iter: int = 200                   # Maximum EM iterations
    tol: float = 1e-4                   # Convergence tolerance
    covariance_type: str = 'diag'       # 'diag' (recommended), 'full', 'spherical', 'tied'
    min_covar: float = 1e-3             # Minimum covariance (regularization floor)
    init_params: str = 'stmc'           # Initialize: startprob, transmat, means, covars
    params: str = 'stmc'                # Parameters to update during EM

    # Sticky priors
    kappa: float = 10.0                 # Sticky kappa (1-50, higher = stickier)
    sticky_alpha: float = 1.0           # Dirichlet concentration for self-transition
    use_sticky_priors: bool = True      # Enable sticky HMM
    post_fit_regularization: bool = True  # Regularize transmat after fitting

    # Initialization
    kmeans_init: bool = True            # Use KMeans++ for initialization
    kmeans_n_init: int = 10             # Number of KMeans initializations
    random_state: Optional[int] = 42    # Random seed

    # Numerical stability
    eigenvalue_floor: float = 1e-6      # Floor for eigenvalues (covariance regularization)
    verbose: bool = False               # Verbose output from hmmlearn


class StickyHMMModel:
    """
    Sticky Hidden Markov Model for regime discovery.

    Implements sticky priors to encourage persistent regimes and includes
    regularization techniques for numerical stability on Mac M1.
    """

    def __init__(self, config: StickyHMMConfig):
        """
        Initialize Sticky HMM model.

        Args:
            config: Sticky HMM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            init_params=config.init_params,
            params=config.params,
            random_state=config.random_state,
            verbose=config.verbose
        )

        # Set minimum covariance for regularization
        self.model.min_covar = config.min_covar

        # Track fitting status
        self.is_fitted = False
        self.convergence_monitor = []
        self.feature_dim = None

    def fit(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> 'StickyHMMModel':
        """
        Fit Sticky HMM model to data.

        Args:
            X: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            Self (fitted model)
        """
        tprint_info(f"🔧 Fitting Sticky HMM (n_components={self.config.n_components}, kappa={self.config.kappa})")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.feature_dim = X.shape[1]
        n_samples = X.shape[0]

        # Step 1: KMeans initialization
        if self.config.kmeans_init:
            tprint_info("  → Initializing with KMeans++")
            self._initialize_with_kmeans(X)

        # Step 2: Initialize transition matrix with sticky priors
        if self.config.use_sticky_priors:
            tprint_info(f"  → Setting sticky priors (kappa={self.config.kappa})")
            self._initialize_sticky_transmat()

        # Step 3: Fit model
        tprint_info(f"  → Running EM algorithm (max_iter={self.config.n_iter})")

        try:
            self.model.fit(X, lengths=lengths)
            self.is_fitted = True

            # Monitor convergence
            if hasattr(self.model, 'monitor_'):
                self.convergence_monitor = self.model.monitor_.history
                converged = self.model.monitor_.converged

                if converged:
                    tprint_info(f"    ✅ Converged after {len(self.convergence_monitor)} iterations")
                else:
                    tprint_warning(f"    ⚠️  Did not converge (max_iter={self.config.n_iter})")

        except Exception as e:
            tprint_error(f"❌ HMM fitting failed: {e}")
            raise

        # Step 4: Post-fit regularization
        if self.config.post_fit_regularization:
            tprint_info("  → Applying post-fit regularization")
            self._regularize_transmat()

        # Step 5: Regularize covariances
        self._regularize_covariances()

        tprint_info(f"✅ Sticky HMM fitted successfully")

        return self

    def predict(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict hidden state sequence using Viterbi algorithm.

        Args:
            X: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            Hidden state sequence (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X, lengths=lengths)

    def predict_proba(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict state probabilities (posterior probabilities).

        Args:
            X: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            State probabilities (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict_proba(X, lengths=lengths)

    def score(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> float:
        """
        Compute log-likelihood of data under the model.

        Args:
            X: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            Log-likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.score(X, lengths=lengths)

    def _initialize_with_kmeans(self, X: np.ndarray):
        """Initialize HMM parameters using KMeans++."""
        kmeans = KMeans(
            n_clusters=self.config.n_components,
            init='k-means++',
            n_init=self.config.kmeans_n_init,
            random_state=self.config.random_state
        )

        labels = kmeans.fit_predict(X)

        # Initialize means from KMeans centroids
        self.model.means_ = kmeans.cluster_centers_

        # Initialize covariances from cluster scatter
        covars = []
        for i in range(self.config.n_components):
            cluster_data = X[labels == i]
            if len(cluster_data) > 1:
                if self.config.covariance_type == 'diag':
                    cov = np.var(cluster_data, axis=0) + self.config.min_covar
                elif self.config.covariance_type == 'full':
                    cov = np.cov(cluster_data.T) + np.eye(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'spherical':
                    cov = np.var(cluster_data) + self.config.min_covar
                else:  # tied
                    cov = np.cov(X.T) + np.eye(self.feature_dim) * self.config.min_covar
            else:
                # Fallback: use global variance
                if self.config.covariance_type == 'diag':
                    cov = np.var(X, axis=0) + self.config.min_covar
                elif self.config.covariance_type == 'full':
                    cov = np.cov(X.T) + np.eye(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'spherical':
                    cov = np.var(X) + self.config.min_covar
                else:  # tied
                    cov = np.cov(X.T) + np.eye(self.feature_dim) * self.config.min_covar

            covars.append(cov)

        if self.config.covariance_type == 'diag':
            self.model.covars_ = np.array(covars)
        elif self.config.covariance_type == 'full':
            self.model.covars_ = np.array(covars)
        elif self.config.covariance_type == 'spherical':
            self.model.covars_ = np.array(covars)
        else:  # tied
            self.model.covars_ = covars[0]

        # Initialize start probabilities from cluster frequencies
        unique, counts = np.unique(labels, return_counts=True)
        startprob = np.zeros(self.config.n_components)
        for state, count in zip(unique, counts):
            startprob[state] = count / len(labels)

        # Add small probability to avoid zero probabilities
        startprob = startprob + 1e-3
        startprob = startprob / startprob.sum()
        self.model.startprob_ = startprob

    def _initialize_sticky_transmat(self):
        """
        Initialize transition matrix with sticky priors.

        Sticky HMM encourages self-transitions by adding kappa to the diagonal
        of the Dirichlet prior.
        """
        n = self.config.n_components

        # Base Dirichlet concentration
        alpha = self.config.sticky_alpha

        # Create transition matrix with sticky diagonal
        # Each row is sampled from Dirichlet(alpha + kappa * I)
        transmat = np.zeros((n, n))

        for i in range(n):
            # Dirichlet parameters: alpha for off-diagonal, alpha + kappa for diagonal
            dirichlet_params = np.full(n, alpha)
            dirichlet_params[i] += self.config.kappa  # Add kappa to diagonal

            # Sample from Dirichlet (or use expected value)
            transmat[i, :] = dirichlet_params / dirichlet_params.sum()

        # Ensure rows sum to 1
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

        self.model.transmat_ = transmat

    def _regularize_transmat(self):
        """
        Post-fit regularization of transition matrix.

        Applies sticky regularization to the learned transition matrix by
        adding kappa to the diagonal and re-normalizing.
        """
        if not hasattr(self.model, 'transmat_'):
            return

        transmat = self.model.transmat_.copy()
        n = self.config.n_components

        # Add kappa to diagonal
        for i in range(n):
            transmat[i, i] += self.config.kappa

        # Re-normalize rows
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

        self.model.transmat_ = transmat

    def _regularize_covariances(self):
        """
        Regularize covariances by adding floor to eigenvalues.

        This prevents singular covariance matrices and improves numerical stability.
        """
        if not hasattr(self.model, 'covars_'):
            return

        if self.config.covariance_type == 'diag':
            # For diagonal covariance, simply add floor
            self.model.covars_ = np.maximum(
                self.model.covars_,
                self.config.eigenvalue_floor
            )

        elif self.config.covariance_type == 'full':
            # For full covariance, regularize eigenvalues
            for i in range(self.config.n_components):
                cov = self.model.covars_[i]

                # Eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Floor eigenvalues
                eigenvalues = np.maximum(eigenvalues, self.config.eigenvalue_floor)

                # Reconstruct covariance matrix
                self.model.covars_[i] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        elif self.config.covariance_type == 'spherical':
            # For spherical covariance, add floor
            self.model.covars_ = np.maximum(
                self.model.covars_,
                self.config.eigenvalue_floor
            )

        elif self.config.covariance_type == 'tied':
            # For tied covariance, regularize eigenvalues
            cov = self.model.covars_

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Floor eigenvalues
            eigenvalues = np.maximum(eigenvalues, self.config.eigenvalue_floor)

            # Reconstruct covariance matrix
            self.model.covars_ = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.transmat_

    def get_state_means(self) -> np.ndarray:
        """Get the learned state means."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.means_

    def get_state_covariances(self) -> np.ndarray:
        """Get the learned state covariances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.covars_

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calculate stationary distribution of the Markov chain.

        The stationary distribution is the left eigenvector corresponding
        to eigenvalue 1.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        transmat = self.model.transmat_

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))

        # Get corresponding eigenvector
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1
        stationary = stationary / stationary.sum()

        return stationary

    def get_expected_durations(self) -> np.ndarray:
        """
        Calculate expected duration in each state.

        Expected duration = 1 / (1 - p_ii) where p_ii is self-transition probability.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        transmat = self.model.transmat_
        self_transition_probs = np.diag(transmat)

        # Expected duration = 1 / (1 - p_ii)
        expected_durations = 1.0 / (1.0 - self_transition_probs + 1e-8)

        return expected_durations

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if not self.is_fitted:
            return {'fitted': False}

        summary = {
            'fitted': True,
            'n_components': self.config.n_components,
            'n_features': self.feature_dim,
            'covariance_type': self.config.covariance_type,
            'kappa': self.config.kappa,
            'min_covar': self.config.min_covar,
            'transition_matrix': self.model.transmat_.tolist(),
            'stationary_distribution': self.get_stationary_distribution().tolist(),
            'expected_durations': self.get_expected_durations().tolist(),
            'convergence_iterations': len(self.convergence_monitor),
            'converged': self.model.monitor_.converged if hasattr(self.model, 'monitor_') else None
        }

        return summary
