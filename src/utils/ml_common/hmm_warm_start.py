"""
HMM Warm Start Helper.

Provides utilities for warm-starting HMM training to achieve faster convergence
during retraining cycles.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from hmmlearn.hmm import GaussianHMM
import logging

logger = logging.getLogger(__name__)


class HMMWarmStarter:
    """
    Manages warm start initialization for HMM models.

    Warm starting uses parameters from a previous HMM as initial values
    for training a new HMM, leading to faster convergence.
    """

    def __init__(self):
        """Initialize HMM warm starter."""
        self.previous_params = None

    def extract_params(self, hmm: GaussianHMM) -> Dict[str, np.ndarray]:
        """
        Extract parameters from a trained HMM.

        Args:
            hmm: Trained HMM model

        Returns:
            Dictionary of HMM parameters
        """
        params = {}

        if hasattr(hmm, 'startprob_'):
            params['startprob_'] = hmm.startprob_.copy()

        if hasattr(hmm, 'transmat_'):
            params['transmat_'] = hmm.transmat_.copy()

        if hasattr(hmm, 'means_'):
            params['means_'] = hmm.means_.copy()

        if hasattr(hmm, 'covars_'):
            params['covars_'] = hmm.covars_.copy()

        logger.info(
            f"Extracted HMM parameters: {hmm.n_components} states, "
            f"{hmm.n_features} features"
        )

        return params

    def create_warm_started_hmm(
        self,
        n_components: int,
        n_features: int,
        previous_hmm: Optional[GaussianHMM] = None,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ) -> GaussianHMM:
        """
        Create HMM with warm start from previous model.

        Args:
            n_components: Number of hidden states
            n_features: Number of features
            previous_hmm: Optional previous HMM to use for warm start
            covariance_type: Type of covariance ('full', 'diag', 'spherical', 'tied')
            n_iter: Maximum iterations
            random_state: Random seed

        Returns:
            GaussianHMM ready for fitting
        """
        # Create new HMM
        hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            init_params='',  # Don't initialize automatically, we'll do it manually
            params='stmc'  # Learn startprob, transmat, means, covars
        )

        def _expected_covars_shape(cov_type: str) -> Optional[Tuple[int, ...]]:
            cov_type_norm = str(cov_type).lower()
            if cov_type_norm == "full":
                return (n_components, n_features, n_features)
            if cov_type_norm == "diag":
                return (n_components, n_features)
            if cov_type_norm == "tied":
                return (n_features, n_features)
            if cov_type_norm == "spherical":
                return (n_components,)
            return None

        # Initialize from previous model if available and compatible
        if previous_hmm is not None:
            if (previous_hmm.n_components == n_components and
                    previous_hmm.n_features == n_features):

                # Copy start probabilities
                if hasattr(previous_hmm, 'startprob_'):
                    hmm.startprob_ = previous_hmm.startprob_.copy()

                # Copy transition matrix
                if hasattr(previous_hmm, 'transmat_'):
                    hmm.transmat_ = previous_hmm.transmat_.copy()

                # Copy means
                if hasattr(previous_hmm, 'means_'):
                    hmm.means_ = previous_hmm.means_.copy()

                # Copy covariances (if same type)
                if (hasattr(previous_hmm, 'covars_') and
                        previous_hmm.covariance_type == covariance_type):
                    expected_shape = _expected_covars_shape(covariance_type)
                    covars = getattr(previous_hmm, 'covars_', None)
                    covars_shape = getattr(covars, 'shape', None)
                    if expected_shape is None:
                        logger.warning(
                            "Unknown covariance_type '%s' for warm-start; skipping covars copy",
                            covariance_type,
                        )
                        hmm.init_params = 'c'
                    elif covars_shape != expected_shape:
                        logger.warning(
                            "Warm-start covars shape mismatch for covariance_type='%s': expected=%s got=%s. "
                            "Skipping covars warm-start and allowing hmmlearn to initialize covariances.",
                            covariance_type,
                            expected_shape,
                            covars_shape,
                        )
                        hmm.init_params = 'c'
                    else:
                        hmm.covars_ = covars.copy()

                logger.info(
                    f"HMM warm-started from previous model "
                    f"({n_components} states, {n_features} features)"
                )

                # If any parameters were not supplied, ensure hmmlearn initializes them.
                # NOTE: init_params controls which parameters are re-initialized before fitting.
                # We keep it empty unless we explicitly detected missing/mismatched covariances.
                if hmm.init_params == '':
                    hmm.init_params = ''
            else:
                logger.warning(
                    f"Previous HMM incompatible "
                    f"(states: {previous_hmm.n_components} vs {n_components}, "
                    f"features: {previous_hmm.n_features} vs {n_features}). "
                    f"Using random initialization."
                )
                # Initialize with default method
                hmm.init_params = 'stmc'

        else:
            # No previous model, use default initialization
            hmm.init_params = 'stmc'

        return hmm

    def validate_hmm_convergence(
        self,
        hmm: GaussianHMM,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate HMM convergence and quality.

        Args:
            hmm: Trained HMM model
            X: Training data

        Returns:
            Dictionary of validation metrics
        """
        metrics = {}

        # Check convergence
        metrics['converged'] = hasattr(hmm, 'monitor_') and hmm.monitor_.converged
        metrics['n_iter'] = hmm.monitor_.iter if hasattr(hmm, 'monitor_') else hmm.n_iter

        # Calculate log-likelihood
        try:
            log_likelihood = hmm.score(X)
            metrics['log_likelihood'] = log_likelihood
            metrics['log_likelihood_per_sample'] = log_likelihood / len(X)
        except Exception as e:
            logger.warning(f"Failed to compute log-likelihood: {e}")
            metrics['log_likelihood'] = None

        # Check transition matrix properties
        if hasattr(hmm, 'transmat_'):
            # Self-transition probabilities (diagonal)
            self_transitions = np.diag(hmm.transmat_)
            metrics['mean_self_transition'] = np.mean(self_transitions)
            metrics['min_self_transition'] = np.min(self_transitions)

            # Check for absorbing states (self-transition = 1.0)
            absorbing_states = np.sum(self_transitions > 0.99)
            metrics['absorbing_states'] = absorbing_states

        # Check state usage
        try:
            state_sequence = hmm.predict(X)
            unique_states = np.unique(state_sequence)
            metrics['states_used'] = len(unique_states)
            metrics['states_unused'] = hmm.n_components - len(unique_states)

            # State distribution
            state_counts = np.bincount(state_sequence, minlength=hmm.n_components)
            state_probs = state_counts / len(state_sequence)
            metrics['state_entropy'] = -np.sum(
                state_probs[state_probs > 0] * np.log(state_probs[state_probs > 0])
            )
            metrics['state_balance'] = metrics['state_entropy'] / np.log(hmm.n_components)

        except Exception as e:
            logger.warning(f"Failed to compute state usage: {e}")

        return metrics

    def compare_hmms(
        self,
        old_hmm: GaussianHMM,
        new_hmm: GaussianHMM
    ) -> Dict[str, float]:
        """
        Compare two HMMs to measure how much they changed.

        Args:
            old_hmm: Previous HMM
            new_hmm: New HMM

        Returns:
            Dictionary of comparison metrics
        """
        if old_hmm.n_components != new_hmm.n_components:
            raise ValueError("Cannot compare HMMs with different number of states")

        metrics = {}

        # Compare transition matrices (Frobenius norm)
        if hasattr(old_hmm, 'transmat_') and hasattr(new_hmm, 'transmat_'):
            trans_diff = np.linalg.norm(old_hmm.transmat_ - new_hmm.transmat_, 'fro')
            metrics['transition_matrix_diff'] = trans_diff

        # Compare means (average Euclidean distance)
        if hasattr(old_hmm, 'means_') and hasattr(new_hmm, 'means_'):
            mean_diffs = []
            for i in range(old_hmm.n_components):
                diff = np.linalg.norm(old_hmm.means_[i] - new_hmm.means_[i])
                mean_diffs.append(diff)
            metrics['mean_state_means_diff'] = np.mean(mean_diffs)
            metrics['max_state_means_diff'] = np.max(mean_diffs)

        # Compare start probabilities
        if hasattr(old_hmm, 'startprob_') and hasattr(new_hmm, 'startprob_'):
            start_diff = np.linalg.norm(old_hmm.startprob_ - new_hmm.startprob_)
            metrics['start_prob_diff'] = start_diff

        return metrics


def create_hmm_with_gmm_init(
    n_components: int,
    n_features: int,
    gmm_means: np.ndarray,
    gmm_covariances: np.ndarray,
    covariance_type: str = "full",
    n_iter: int = 100,
    random_state: int = 42
) -> GaussianHMM:
    """
    Create HMM initialized from GMM clustering.

    This provides a good initialization when training an HMM from scratch,
    using GMM to identify initial state means and covariances.

    Args:
        n_components: Number of hidden states
        n_features: Number of features
        gmm_means: GMM component means (n_components, n_features)
        gmm_covariances: GMM component covariances
        covariance_type: Type of covariance
        n_iter: Maximum iterations
        random_state: Random seed

    Returns:
        GaussianHMM with GMM initialization
    """
    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params='st',  # Only initialize start_prob and transmat
        params='stmc'
    )

    # Set means from GMM
    hmm.means_ = gmm_means.copy()

    # Set covariances from GMM
    if covariance_type == 'full' and len(gmm_covariances.shape) == 3:
        hmm.covars_ = gmm_covariances.copy()
    elif covariance_type == 'diag':
        if len(gmm_covariances.shape) == 3:
            # Extract diagonals from full covariances
            hmm.covars_ = np.array([np.diag(cov) for cov in gmm_covariances])
        else:
            hmm.covars_ = gmm_covariances.copy()
    elif covariance_type == 'spherical':
        if len(gmm_covariances.shape) == 3:
            # Average diagonal elements
            hmm.covars_ = np.array([np.mean(np.diag(cov)) for cov in gmm_covariances])
        else:
            hmm.covars_ = gmm_covariances.copy()

    logger.info(
        f"HMM initialized from GMM clustering "
        f"({n_components} states, {n_features} features)"
    )

    return hmm
