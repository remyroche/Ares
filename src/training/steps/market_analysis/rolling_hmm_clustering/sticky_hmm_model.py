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

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, TypeVar, Literal
import logging
import warnings

import numpy.typing as npt
from hmmlearn import hmm
from hmmlearn.base import BaseHMM
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array

# Type aliases
ArrayLike = Union[npt.NDArray[np.float64], pd.Series, pd.DataFrame]
Numeric = Union[int, float, np.number]
RandomState = Optional[Union[int, np.random.RandomState]]
CovarianceType = Literal['spherical', 'tied', 'diag', 'full']

# Type variable for generic HMM type
HMMType = TypeVar('HMMType', bound=BaseHMM)

logger = logging.getLogger(__name__)

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class StickyHMMConfig:
    """Configuration for Sticky HMM model."""
    
    def __init__(
        self,
        n_components: int = 5,               # Number of hidden states (4-6 recommended)
        n_iter: int = 200,                   # Maximum EM iterations
        tol: float = 1e-4,                   # Convergence tolerance
        covariance_type: CovarianceType = 'diag',  # 'diag' (recommended), 'full', 'spherical', 'tied'
        min_covar: float = 1e-3,             # Minimum covariance (regularization floor)
        init_params: str = 'stmc',           # Initialize: startprob, transmat, means, covars
        params: str = 'stmc',                # Parameters to update during EM
        
        # Sticky priors
        kappa: float = 10.0,                 # Sticky kappa (1-50, higher = stickier)
        sticky_alpha: float = 1.0,           # Dirichlet concentration for self-transition
        use_sticky_priors: bool = True,      # Enable sticky HMM
        post_fit_regularization: bool = True,  # Regularize transmat after fitting
        
        # Initialization
        kmeans_init: bool = True,            # Use KMeans++ for initialization
        kmeans_n_init: int = 10,             # Number of KMeans initializations
        random_state: RandomState = 42,      # Random seed
        
        # Numerical stability
        eigenvalue_floor: float = 1e-6,      # Floor for eigenvalues (covariance regularization)
        verbose: bool = False                # Verbose output from hmmlearn
    ) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.init_params = init_params
        self.params = params
        self.kappa = kappa
        self.sticky_alpha = sticky_alpha
        self.use_sticky_priors = use_sticky_priors
        self.post_fit_regularization = post_fit_regularization
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state
        self.eigenvalue_floor = eigenvalue_floor
        self.verbose = verbose
        
        # Validate parameters
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate configuration parameters."""
        if self.n_components < 2:
            raise ValueError(f"n_components must be >= 2, got {self.n_components}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        if self.sticky_alpha <= 0:
            raise ValueError(f"sticky_alpha must be > 0, got {self.sticky_alpha}")
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError(f"Invalid covariance_type: {self.covariance_type}")


class StickyHMMModel:
    """
    Sticky Hidden Markov Model for regime discovery.

    Implements sticky priors to encourage persistent regimes and includes
    regularization techniques for numerical stability on Mac M1.
    """

    def __init__(self, config: StickyHMMConfig) -> None:
        """
        Initialize Sticky HMM model.

        Args:
            config: Sticky HMM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model state
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None
        self.feature_dim: int = 0  # Initialize with default value
        self._model: Optional[hmm.GaussianHMM] = None
        
        # Initialize HMM model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying HMM model with current configuration."""
        logger.debug(
            "Initializing StickyHMMModel",
            f"n_components={self.config.n_components}",
            f"covariance_type={self.config.covariance_type}",
            f"kappa={self.config.kappa}"
        )
        
        self._model = hmm.GaussianHMM(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            init_params=self.config.init_params,
            params=self.config.params,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
        
        # Set minimum covariance to avoid numerical issues
        if hasattr(self._model, 'min_covar'):
            self._model.min_covar = self.config.min_covar

        # Set minimum covariance for regularization
        self.model.min_covar = self.config.min_covar

        # Track fitting status
        self._is_fitted = False
        self.convergence_monitor = []
        self.feature_dim = None

    @property
    def model(self) -> hmm.GaussianHMM:
        """Get the underlying HMM model."""
        if self._model is None:
            raise RuntimeError("Model has not been initialized")
        return self._model
    
    @property
    def n_features(self) -> int:
        """Get the number of features in the model."""
        if self._n_features is None:
            raise RuntimeError("Number of features not set. Fit the model first.")
        return self._n_features
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted
    
    def fit(self, x: ArrayLike, lengths: Optional[np.ndarray] = None) -> 'StickyHMMModel':
        """
        Fit the Sticky HMM to the data.

        Args:
            x: Input data of shape (n_samples, n_features)
            lengths: Lengths of the individual sequences in x

        Returns:
            self: The fitted model
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model initialization fails
        """
        logger.info("🔧 Fitting Sticky HMM model...")
        
        # Convert input to numpy array if needed
        x_array = self._validate_and_convert_input(x)
        self._n_features = x_array.shape[1]
        self.feature_dim = x_array.shape[1]
        
        # Initialize model parameters with sticky priors if enabled
        if self.config.use_sticky_priors:
            self._initialize_with_sticky_priors(x_array)
            if getattr(self.model, "init_params", None):
                adjusted = ''.join(ch for ch in self.model.init_params if ch not in {'s', 't', 'm', 'c'})
                if adjusted != self.model.init_params:
                    logger.debug(
                        "Stripping init_params '%s' -> '%s' to preserve manual initialization",
                        self.model.init_params,
                        adjusted
                    )
                    self.model.init_params = adjusted

        # Step 3: Fit model
        logger.info(f"  → Running EM algorithm (max_iter={self.config.n_iter})")

        try:
            self.model.fit(x_array, lengths=lengths)
            self._is_fitted = True

            # Monitor convergence
            if hasattr(self.model, 'monitor_'):
                self.convergence_monitor = self.model.monitor_.history
                converged = self.model.monitor_.converged

                if converged:
                    logger.info(f"    ✅ Converged after {len(self.convergence_monitor)} iterations")
                else:
                    logger.warning(f"    ⚠️  Did not converge (max_iter={self.config.n_iter})")

        except Exception as e:
            logger.error(f"❌ HMM fitting failed: {e}")
            raise

        # Step 4: Post-fit regularization
        if self.config.post_fit_regularization:
            logger.info("  → Applying post-fit regularization")
            self._regularize_transmat()

        # Step 5: Regularize covariances
        self._regularize_covariances()

        logger.info(f"✅ Sticky HMM fitted successfully")

        return self

    def predict(self, x: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict hidden state sequence using Viterbi algorithm.

        Args:
            x: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            Hidden state sequence (n_samples,)
        """
        logger.debug("Predicting hidden states with StickyHMMModel")
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.model.predict(x, lengths=lengths)

    def predict_proba(self, x: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict state probabilities (posterior probabilities).

        Args:
            x: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            State probabilities (n_samples, n_components)
        """
        logger.debug("Computing posterior probabilities with StickyHMMModel")
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.model.predict_proba(x, lengths=lengths)

    def score(self, x: np.ndarray, lengths: Optional[np.ndarray] = None) -> float:
        """
        Compute log-likelihood of the data under the model.

        Args:
            x: Feature data (n_samples, n_features)
            lengths: Sequence lengths (for multiple sequences)

        Returns:
            Log-likelihood
        """
        logger.debug("Scoring data under StickyHMMModel")
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return self.model.score(x, lengths=lengths)

    def _initialize_with_kmeans(self, x: np.ndarray):
        """Initialize HMM parameters using KMeans++."""
        logger.debug("Initializing StickyHMMModel parameters with KMeans++")
        kmeans = KMeans(
            n_clusters=self.config.n_components,
            init='k-means++',
            n_init=str(self.config.kmeans_n_init),
            random_state=self.config.random_state
        )

        labels = kmeans.fit_predict(x)

        # Initialize means from KMeans centroids
        self.model.means_ = kmeans.cluster_centers_

        # Initialize covariances from cluster scatter
        covars = []
        for i in range(self.config.n_components):
            cluster_data = x[labels == i]
            if len(cluster_data) > 1:
                if self.config.covariance_type == 'diag':
                    cov = np.var(cluster_data, axis=0) + self.config.min_covar
                elif self.config.covariance_type == 'full':
                    cov = np.cov(cluster_data.T) + np.eye(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'spherical':
                    cov = np.var(cluster_data) + self.config.min_covar
                else:  # tied
                    cov = np.cov(x.T) + np.eye(self.feature_dim) * self.config.min_covar
            else:
                # Fallback: use global variance
                if self.config.covariance_type == 'diag':
                    cov = np.var(x, axis=0) + self.config.min_covar
                elif self.config.covariance_type == 'full':
                    cov = np.cov(x.T) + np.eye(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'spherical':
                    cov = np.var(x) + self.config.min_covar
                else:  # tied
                    cov = np.cov(x.T) + np.eye(self.feature_dim) * self.config.min_covar

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
        logger.debug("Applying sticky prior initialization to transition matrix")
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
        logger.debug("Regularizing transition matrix with sticky priors")
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

    def _flatten_covariances_for_logging(self) -> Dict[str, Any]:
        """Return covariance shapes for diagnostics without mutating state."""
        covars = getattr(self.model, "covars_", None)
        if covars is None:
            return {"present": False}

        covars_array = np.asarray(covars, dtype=float)
        shape = covars_array.shape
        summary: Dict[str, Any] = {
            "present": True,
            "raw_shape": shape,
            "dtype": str(covars_array.dtype)
        }

        if covars_array.ndim == 3:
            summary["matrix_shapes"] = [covars_array[i].shape for i in range(min(3, covars_array.shape[0]))]
        return summary

    def _ensure_diag_covariance_shape(self, covars: np.ndarray) -> np.ndarray:
        """Ensure diagonal covariances have shape (n_components, n_features)."""
        if covars is None:
            raise ValueError("Covariance array cannot be None when regularizing")

        covars_arr = np.asarray(covars, dtype=float)
        n_components = self.config.n_components

        # Determine feature dimension from known attributes when available
        feature_dim: Optional[int] = self.feature_dim
        if feature_dim is None and hasattr(self.model, "means_"):
            feature_dim = int(self.model.means_.shape[1])

        if covars_arr.ndim == 3:
            if covars_arr.shape[0] != n_components:
                raise ValueError(
                    f"Diagonal covariance array shape {covars_arr.shape} incompatible with n_components={n_components}"
                )
            if covars_arr.shape[1] != covars_arr.shape[2]:
                raise ValueError(
                    f"Expected square covariance matrices for diag extraction, got shape {covars_arr.shape}"
                )
            covars_arr = np.array([np.diag(covars_arr[i]) for i in range(n_components)])

        if covars_arr.ndim == 1:
            total = covars_arr.size
            if feature_dim is None and total % n_components == 0:
                feature_dim = int(total // n_components)
            if feature_dim is None or feature_dim <= 0:
                raise ValueError(
                    "Unable to infer feature dimension for diagonal covariance regularization"
                )
            if feature_dim * n_components != total:
                raise ValueError(
                    f"Diagonal covariance size {total} incompatible with n_components={n_components}"
                )
            covars_arr = covars_arr.reshape(n_components, feature_dim)

        elif covars_arr.ndim == 2:
            if covars_arr.shape[0] != n_components:
                if covars_arr.size % n_components != 0:
                    raise ValueError(
                        f"Diagonal covariance array shape {covars_arr.shape} incompatible with n_components={n_components}"
                    )
                covars_arr = covars_arr.reshape(n_components, -1)

        else:
            covars_arr = covars_arr.reshape(n_components, -1)

        if feature_dim is None:
            # Infer from reshaped array if still unknown
            feature_dim_candidate = covars_arr.shape[1] if covars_arr.ndim == 2 else None
            if feature_dim_candidate is None or feature_dim_candidate <= 0:
                raise ValueError(
                    "Unable to infer feature dimension after covariance reshaping"
                )
            feature_dim = int(feature_dim_candidate)

        if covars_arr.shape[1] != feature_dim:
            if covars_arr.shape[1] == 1 and feature_dim == 1:
                return covars_arr
            if feature_dim > 0 and covars_arr.shape[1] % feature_dim == 0:
                collapse_factor = int(covars_arr.shape[1] // feature_dim)
                logger.debug(
                    "Collapsing diagonal covariances from shape %s with factor %d",
                    covars_arr.shape,
                    collapse_factor
                )
                covars_arr = covars_arr.reshape(n_components, collapse_factor, feature_dim).mean(axis=1)
            elif covars_arr.size % (n_components * feature_dim) == 0 and feature_dim > 0:
                collapse_factor = int(covars_arr.size // (n_components * feature_dim))
                covars_arr = covars_arr.reshape(n_components, collapse_factor, feature_dim).mean(axis=1)
            elif covars_arr.size == n_components * feature_dim:
                logger.debug(
                    "Reshaping diagonal covariances from flat array of size %d into (%d, %d)",
                    covars_arr.size,
                    n_components,
                    feature_dim
                )
                covars_arr = covars_arr.reshape(n_components, feature_dim)
            else:
                raise ValueError(
                    f"Diagonal covariance shape {covars_arr.shape} does not match expected {(n_components, feature_dim)}"
                )

        # Persist inferred feature dimension for subsequent calls
        if feature_dim is not None and feature_dim > 0:
            self.feature_dim = feature_dim

        return covars_arr

    def _regularize_covariances(self):
        """
        Regularize covariances by adding floor to eigenvalues.

        This prevents singular covariance matrices and improves numerical stability.
        """
        logger.debug("Regularizing covariance matrices for numerical stability")
        if not hasattr(self.model, 'covars_'):
            return

        if self.config.covariance_type == 'diag':
            # For diagonal covariance, make sure we maintain (n_components, n_features)
            if self.model.covars_ is not None:
                covar_debug = self._flatten_covariances_for_logging()
                logger.debug("Covariance diagnostics before regularization: %s", covar_debug)
                covars = self._ensure_diag_covariance_shape(self.model.covars_)
                covars = np.maximum(covars, self.config.eigenvalue_floor)
                logger.debug("Covariance diagnostics after reshape: shape=%s", covars.shape)
                self.model.covars_ = covars

        elif self.config.covariance_type == 'full':
            # For full covariance, regularize eigenvalues
            if self.model.covars_ is not None:
                for i in range(self.config.n_components):
                    cov = self.model.covars_[i]
                    if cov is not None:
                        # Eigenvalue decomposition
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)

                        # Floor eigenvalues
                        eigenvalues = np.maximum(eigenvalues, self.config.eigenvalue_floor)

                        # Reconstruct covariance matrix
                        self.model.covars_[i] = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        elif self.config.covariance_type == 'spherical':
            # For spherical covariance, add floor
            if self.model.covars_ is not None:
                self.model.covars_ = np.maximum(
                    self.model.covars_,
                    self.config.eigenvalue_floor
                )

        elif self.config.covariance_type == 'tied':
            # For tied covariance, regularize eigenvalues
            cov = self.model.covars_
            if cov is not None:
                # Eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Floor eigenvalues
                eigenvalues = np.maximum(eigenvalues, self.config.eigenvalue_floor)

                # Reconstruct covariance matrix
                self.model.covars_ = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix."""
        logger.debug("Retrieving HMM transition matrix")
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.transmat_

    def get_state_means(self) -> np.ndarray:
        """Get the learned state means."""
        logger.debug("Retrieving HMM state means")
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.means_

    def get_state_covariances(self) -> np.ndarray:
        """Get the learned state covariances."""
        logger.debug("Retrieving HMM state covariances")
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.covars_

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Calculate stationary distribution of the Markov chain.

        The stationary distribution is the left eigenvector corresponding
        to eigenvalue 1.
        """
        logger.debug("Computing stationary distribution for StickyHMMModel")
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
        logger.debug("Calculating expected state durations for StickyHMMModel")
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        transmat = self.model.transmat_
        self_transition_probs = np.diag(transmat)

        # Expected duration = 1 / (1 - p_ii)
        expected_durations = 1.0 / (1.0 - self_transition_probs + 1e-8)

        return expected_durations

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        logger.debug("Building summary for StickyHMMModel")
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

    def _validate_and_convert_input(self, X: ArrayLike) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Args:
            X: Input data of various types
            
        Returns:
            Validated numpy array
        """
        if X is None:
            raise ValueError("Input data cannot be None")
            
        X_array = check_array(X, ensure_2d=True, dtype='float64')
        return X_array
    
    def _initialize_with_sticky_priors(self, X: np.ndarray) -> None:
        """
        Initialize model with sticky priors for better regime persistence.
        
        Args:
            X: Input data array
        """
        logger.debug("Initializing with sticky priors")
        
        # Use K-means to get initial state assignments
        kmeans = KMeans(n_clusters=self.config.n_components, 
                       n_init='auto', random_state=self.config.random_state)
        labels = kmeans.fit_predict(X)
        
        # Initialize means based on K-means centroids
        self.model.means_ = kmeans.cluster_centers_
        
        # Initialize covariance matrices
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
                # Fallback if cluster has too few points
                if self.config.covariance_type == 'diag':
                    cov = np.ones(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'full':
                    cov = np.eye(self.feature_dim) * self.config.min_covar
                elif self.config.covariance_type == 'spherical':
                    cov = self.config.min_covar
                else:  # tied
                    cov = np.eye(self.feature_dim) * self.config.min_covar
            
            # Set covariance for this state
            if hasattr(self.model, 'covars_') and self.model.covars_ is not None:
                if self.config.covariance_type == 'diag':
                    self.model.covars_[i] = cov
                elif self.config.covariance_type == 'full':
                    self.model.covars_[i] = cov
                elif self.config.covariance_type == 'spherical':
                    self.model.covars_[i] = cov
                else:  # tied
                    self.model.covars_ = cov
        
        # Initialize transition matrix with sticky priors
        transmat = np.full((self.config.n_components, self.config.n_components), 
                          1.0 / self.config.n_components)
        
        # Add sticky diagonal (higher self-transition probabilities)
        sticky_weight = 0.7  # Probability of staying in same state
        off_diag_weight = (1.0 - sticky_weight) / (self.config.n_components - 1)
        
        for i in range(self.config.n_components):
            transmat[i, i] = sticky_weight
            for j in range(self.config.n_components):
                if i != j:
                    transmat[i, j] = off_diag_weight
        
        # Normalize to ensure rows sum to 1
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        self.model.transmat_ = transmat
        
        # Initialize start probabilities as uniform
        self.model.startprob_ = np.ones(self.config.n_components) / self.config.n_components
