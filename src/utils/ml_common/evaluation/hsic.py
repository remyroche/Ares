
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from typing import Optional, Union

def calculate_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_X: str = 'rbf',
    kernel_Y: str = 'rbf',
    sigma_X: Optional[float] = None,
    sigma_Y: Optional[float] = None,
    subsample: int = 2000,
    random_state: int = 42
) -> float:
    """
    Calculate the Hilbert-Schmidt Independence Criterion (HSIC) between X and Y.

    HSIC measures the dependence between two random variables X and Y.
    It is 0 if and only if X and Y are independent (given characteristic kernels).

    Args:
        X: Input array (n_samples, n_features_X) or (n_samples,)
        Y: Input array (n_samples, n_features_Y) or (n_samples,)
        kernel_X: Kernel for X ('rbf', 'linear', 'delta' for discrete)
        kernel_Y: Kernel for Y ('rbf', 'linear', 'delta' for discrete)
        sigma_X: Gamma parameter for RBF kernel of X (default: 1/n_features)
        sigma_Y: Gamma parameter for RBF kernel of Y (default: 1/n_features)
        subsample: Maximum number of samples to use (default: 2000)
        random_state: Random state for subsampling

    Returns:
        HSIC value (float)
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Handle NaNs: drop rows with NaNs in either X or Y
    mask = ~np.isnan(X).any(axis=1) if X.ndim > 1 else ~np.isnan(X)
    mask &= ~np.isnan(Y).any(axis=1) if Y.ndim > 1 else ~np.isnan(Y)

    X = X[mask]
    Y = Y[mask]

    n_samples = X.shape[0]
    if n_samples < 10:
        return 0.0

    # Subsample if necessary
    if n_samples > subsample:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_samples, subsample, replace=False)
        X = X[indices]
        Y = Y[indices]
        n_samples = subsample

    # Reshape 1D arrays to 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Compute Kernel Matrix K for X
    if kernel_X == 'rbf':
        K = rbf_kernel(X, gamma=sigma_X)
    elif kernel_X == 'linear':
        K = linear_kernel(X)
    elif kernel_X == 'delta':
        # Delta kernel: K_ij = 1 if x_i == x_j else 0 (useful for discrete labels)
        # Using equality check with broadcasting
        K = (X[:, 0][:, None] == X[:, 0][None, :]).astype(float)
    else:
        raise ValueError(f"Unknown kernel_X: {kernel_X}")

    # Compute Kernel Matrix L for Y
    if kernel_Y == 'rbf':
        L = rbf_kernel(Y, gamma=sigma_Y)
    elif kernel_Y == 'linear':
        L = linear_kernel(Y)
    elif kernel_Y == 'delta':
        L = (Y[:, 0][:, None] == Y[:, 0][None, :]).astype(float)
    else:
        raise ValueError(f"Unknown kernel_Y: {kernel_Y}")

    # Center the kernel matrices
    # H = I - 1/n * 1 * 1^T
    # Kc = H * K * H
    # But more efficiently: Kc_ij = K_ij - mean_row_i - mean_col_j + grand_mean

    # Efficient calculation of HSIC = 1/(n-1)^2 * tr(K_centered * L_centered)
    # Actually standard formula is tr(KHLH) / (n-1)^2

    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

    # HSIC = tr(KHLH) / (n-1)^2
    # Note: K and L are symmetric

    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_value = np.trace(Kc @ Lc) / ((n_samples - 1) ** 2)

    return float(hsic_value)
