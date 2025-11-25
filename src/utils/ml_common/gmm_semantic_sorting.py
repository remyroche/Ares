"""
GMM Semantic Sorting to Prevent Label Switching.

Implements sorting and remapping of GMM components to ensure consistent
labeling across retraining iterations, preventing the "label switching" problem.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)


class GMMSemanticSorter:
    """
    Sorts GMM components by semantic meaning to prevent label switching.

    The label switching problem occurs when GMM components are randomly
    ordered after fitting, causing component 0 in one training to correspond
    to component 1 in another training. This breaks consistency for downstream
    models.

    Solution: After fitting GMM, sort components by a semantic criterion
    (e.g., mean of a key feature) and remap labels accordingly.
    """

    def __init__(self, sort_by: str = 'mean_magnitude'):
        """
        Initialize semantic sorter.

        Args:
            sort_by: Sorting criterion:
                - 'mean_magnitude': Sort by L2 norm of component means
                - 'first_feature': Sort by first feature mean
                - 'dominant_feature': Sort by the feature with highest variance
        """
        self.sort_by = sort_by
        self.component_order = None
        self.inverse_order = None

    def fit_and_sort(
        self,
        gmm: GaussianMixture,
        X: np.ndarray,
        sort_feature_idx: Optional[int] = None
    ) -> Tuple[GaussianMixture, np.ndarray]:
        """
        Fit GMM and sort components by semantic criterion.

        Args:
            gmm: Unfitted GaussianMixture model
            X: Training data
            sort_feature_idx: Optional specific feature index for sorting

        Returns:
            Tuple of (sorted_gmm, component_order)
            - sorted_gmm: GMM with sorted components
            - component_order: Array mapping old component IDs to new IDs
        """
        # Fit GMM
        gmm.fit(X)

        # Get sorting criterion
        sort_values = self._compute_sort_values(gmm, sort_feature_idx)

        # Get component order (ascending)
        self.component_order = np.argsort(sort_values)
        self.inverse_order = np.argsort(self.component_order)

        # Remap GMM components
        sorted_gmm = self._remap_gmm_components(gmm, self.component_order)

        logger.info(
            f"GMM components sorted by {self.sort_by}. "
            f"Order: {self.component_order} (values: {sort_values[self.component_order]})"
        )

        return sorted_gmm, self.component_order

    def _compute_sort_values(
        self,
        gmm: GaussianMixture,
        sort_feature_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute values used for sorting components.

        Args:
            gmm: Fitted GaussianMixture model
            sort_feature_idx: Optional specific feature index for sorting

        Returns:
            Array of sort values, one per component
        """
        n_components = gmm.n_components

        if self.sort_by == 'mean_magnitude':
            # Sort by L2 norm of mean vectors
            sort_values = np.linalg.norm(gmm.means_, axis=1)

        elif self.sort_by == 'first_feature':
            # Sort by first feature mean
            sort_values = gmm.means_[:, 0]

        elif self.sort_by == 'dominant_feature':
            # Sort by the feature with highest variance across components
            if sort_feature_idx is None:
                # Find feature with highest variance across component means
                mean_variance = np.var(gmm.means_, axis=0)
                sort_feature_idx = np.argmax(mean_variance)
                logger.info(f"Using feature {sort_feature_idx} for sorting (highest variance)")

            sort_values = gmm.means_[:, sort_feature_idx]

        else:
            # Default: use component index (no sorting)
            sort_values = np.arange(n_components)

        return sort_values

    def _remap_gmm_components(
        self,
        gmm: GaussianMixture,
        component_order: np.ndarray
    ) -> GaussianMixture:
        """
        Create new GMM with remapped components.

        Args:
            gmm: Original fitted GMM
            component_order: Array mapping old to new component IDs

        Returns:
            New GMM with remapped components
        """
        # Create new GMM with same parameters
        sorted_gmm = GaussianMixture(
            n_components=gmm.n_components,
            covariance_type=gmm.covariance_type,
            max_iter=gmm.max_iter,
            random_state=gmm.random_state,
            warm_start=gmm.warm_start
        )

        # Remap component parameters
        sorted_gmm.weights_ = gmm.weights_[component_order]
        sorted_gmm.means_ = gmm.means_[component_order]

        # Remap covariances (structure depends on covariance_type)
        if gmm.covariance_type == 'full':
            sorted_gmm.covariances_ = gmm.covariances_[component_order]
        elif gmm.covariance_type == 'tied':
            sorted_gmm.covariances_ = gmm.covariances_  # Shared covariance
        elif gmm.covariance_type == 'diag':
            sorted_gmm.covariances_ = gmm.covariances_[component_order]
        elif gmm.covariance_type == 'spherical':
            sorted_gmm.covariances_ = gmm.covariances_[component_order]

        # Copy precisions if available
        if hasattr(gmm, 'precisions_'):
            if gmm.covariance_type == 'full':
                sorted_gmm.precisions_ = gmm.precisions_[component_order]
            elif gmm.covariance_type == 'tied':
                sorted_gmm.precisions_ = gmm.precisions_
            elif gmm.covariance_type == 'diag':
                sorted_gmm.precisions_ = gmm.precisions_[component_order]
            elif gmm.covariance_type == 'spherical':
                sorted_gmm.precisions_ = gmm.precisions_[component_order]

        # Copy precisions cholesky if available
        if hasattr(gmm, 'precisions_cholesky_'):
            if gmm.covariance_type == 'full':
                sorted_gmm.precisions_cholesky_ = gmm.precisions_cholesky_[component_order]
            elif gmm.covariance_type == 'tied':
                sorted_gmm.precisions_cholesky_ = gmm.precisions_cholesky_
            elif gmm.covariance_type == 'diag':
                sorted_gmm.precisions_cholesky_ = gmm.precisions_cholesky_[component_order]
            elif gmm.covariance_type == 'spherical':
                sorted_gmm.precisions_cholesky_ = gmm.precisions_cholesky_[component_order]

        # Mark as converged
        sorted_gmm.converged_ = gmm.converged_
        sorted_gmm.n_iter_ = gmm.n_iter_
        sorted_gmm.lower_bound_ = gmm.lower_bound_

        return sorted_gmm

    def remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Remap component labels from old to new ordering.

        Args:
            labels: Array of component labels (old ordering)

        Returns:
            Array of component labels (new ordering)
        """
        if self.inverse_order is None:
            raise ValueError("Must call fit_and_sort before remapping labels")

        # Create mapping array
        remapped = np.zeros_like(labels)
        for old_label, new_label in enumerate(self.inverse_order):
            remapped[labels == old_label] = new_label

        return remapped

    def remap_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Remap component probabilities from old to new ordering.

        Args:
            probabilities: Array of shape (n_samples, n_components) (old ordering)

        Returns:
            Array of shape (n_samples, n_components) (new ordering)
        """
        if self.inverse_order is None:
            raise ValueError("Must call fit_and_sort before remapping probabilities")

        return probabilities[:, self.inverse_order]


def create_warm_started_gmm(
    n_components: int,
    previous_gmm: Optional[GaussianMixture] = None,
    covariance_type: str = 'full',
    max_iter: int = 100,
    random_state: int = 42
) -> GaussianMixture:
    """
    Create GMM with warm start from previous model.

    Args:
        n_components: Number of mixture components
        previous_gmm: Optional previous GMM to use for warm start
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        GaussianMixture model ready for fitting
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=random_state,
        warm_start=True if previous_gmm is not None else False
    )

    # Initialize from previous model if available
    if previous_gmm is not None:
        if previous_gmm.n_components == n_components:
            # Copy parameters for warm start
            if hasattr(previous_gmm, 'weights_'):
                gmm.weights_init = previous_gmm.weights_.copy()
            if hasattr(previous_gmm, 'means_'):
                gmm.means_init = previous_gmm.means_.copy()
            if hasattr(previous_gmm, 'precisions_init'):
                gmm.precisions_init = previous_gmm.precisions_init.copy()

            logger.info("GMM initialized with warm start from previous model")
        else:
            logger.warning(
                f"Previous GMM has {previous_gmm.n_components} components, "
                f"but new GMM has {n_components}. Cannot use warm start."
            )

    return gmm


def measure_gmm_quality(
    gmm: GaussianMixture,
    X: np.ndarray
) -> Dict[str, float]:
    """
    Measure quality metrics for a fitted GMM.

    Args:
        gmm: Fitted GaussianMixture model
        X: Data used for fitting

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Log-likelihood
    metrics['log_likelihood'] = gmm.score(X) * len(X)

    # BIC and AIC
    metrics['bic'] = gmm.bic(X)
    metrics['aic'] = gmm.aic(X)

    # Component separation (mean distance between component means)
    if gmm.n_components > 1:
        mean_dists = []
        for i in range(gmm.n_components):
            for j in range(i + 1, gmm.n_components):
                dist = np.linalg.norm(gmm.means_[i] - gmm.means_[j])
                mean_dists.append(dist)
        metrics['mean_component_separation'] = np.mean(mean_dists)
        metrics['min_component_separation'] = np.min(mean_dists)
    else:
        metrics['mean_component_separation'] = 0.0
        metrics['min_component_separation'] = 0.0

    # Component balance (how evenly distributed are the weights)
    weight_entropy = -np.sum(gmm.weights_ * np.log(gmm.weights_ + 1e-10))
    max_entropy = np.log(gmm.n_components)
    metrics['weight_balance'] = weight_entropy / max_entropy if max_entropy > 0 else 1.0

    # Convergence
    metrics['converged'] = float(gmm.converged_)
    metrics['n_iterations'] = gmm.n_iter_

    return metrics
