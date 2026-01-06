"""
Covariance Matrix Denoising using Marcenko-Pastur Distribution

Implements Marcos López de Prado's approach to removing noise from correlation matrices:
1. Calculate eigenvalues of correlation matrix
2. Identify noise eigenvalues using Marcenko-Pastur distribution
3. Replace noise eigenvalues with constant (shrinkage)
4. Reconstruct denoised correlation matrix

This prevents position sizing from betting on spurious correlations.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Optional
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

def marcenko_pastur_distribution(q: float, sigma: float = 1.0, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Marcenko-Pastur distribution eigenvalue bounds.
    
    The MP distribution describes the eigenvalue distribution of random correlation matrices.
    
    Args:
        q: Ratio of variables to observations (T/N where T=time periods, N=variables)
        sigma: Standard deviation (default: 1.0 for correlation matrices)
        n_points: Number of points for distribution calculation
        
    Returns:
        Tuple of (eigenvalues, pdf_values)
    """
    # MP distribution bounds
    lambda_min = sigma**2 * (1 - np.sqrt(1/q))**2 if q > 1 else 0
    lambda_max = sigma**2 * (1 + np.sqrt(1/q))**2 if q > 1 else 4 * sigma**2
    
    if q <= 1:
        tprint_warning(f"⚠️ q={q:.3f} <= 1, MP distribution not well-defined")
        return np.array([lambda_min, lambda_max]), np.array([0, 1])
    
    # Generate eigenvalue range
    eigenvalues = np.linspace(lambda_min, lambda_max, n_points)
    
    # MP probability density function
    pdf = np.zeros_like(eigenvalues)
    valid_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
    
    if np.any(valid_mask):
        lambda_valid = eigenvalues[valid_mask]
        pdf[valid_mask] = (q / (2 * np.pi * sigma**2 * lambda_valid)) * \
                         np.sqrt((lambda_max - lambda_valid) * (lambda_valid - lambda_min))
    
    return eigenvalues, pdf

def denoise_correlation_mp(
    correlation_matrix: pd.DataFrame, 
    method: str = 'shrink',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Denoise correlation matrix using Marcenko-Pastur distribution.
    
    Args:
        correlation_matrix: Input correlation matrix (DataFrame)
        method: Denoising method ('shrink' or 'filter')
        verbose: Whether to print diagnostic information
        
    Returns:
        Denoised correlation matrix (DataFrame)
    """
    if verbose:
        tprint_info("🔍 Starting Marcenko-Pastur correlation denoising...")
    
    # Validate input
    if not isinstance(correlation_matrix, pd.DataFrame):
        corr_df = pd.DataFrame(correlation_matrix)
    else:
        corr_df = correlation_matrix.copy()
    
    n_assets = corr_df.shape[0]
    if n_assets < 2:
        if verbose:
            tprint_warning("⚠️ Correlation matrix too small (< 2x2), returning original")
        return corr_df
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(corr_df.values)
    
    # Sort in descending order
    idx_desc = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_desc]
    eigenvectors = eigenvectors[:, idx_desc]
    
    # Calculate q ratio (T/N)
    # For correlation matrices, we assume T >> N, use conservative estimate
    q_estimate = min(5.0, max(1.1, 2.0))  # Conservative q between 1.1 and 5.0
    
    if verbose:
        tprint_info(f"📊 Matrix info: {n_assets}x{n_assets}, q≈{q_estimate:.2f}")
        tprint_info(f"📈 Eigenvalue range: {eigenvalues.min():.4f} to {eigenvalues.max():.4f}")
    
    # Calculate Marcenko-Pastur distribution
    mp_eigenvalues, mp_pdf = marcenko_pastur_distribution(q_estimate)
    lambda_min, lambda_max = mp_eigenvalues[0], mp_eigenvalues[-1]
    
    if verbose:
        tprint_info(f"🎯 MP bounds: λ_min={lambda_min:.4f}, λ_max={lambda_max:.4f}")
    
    # Identify noise eigenvalues
    noise_mask = (eigenvalues >= lambda_min) & (eigenvalues <= lambda_max)
    signal_mask = ~noise_mask
    
    n_noise = np.sum(noise_mask)
    n_signal = np.sum(signal_mask)
    
    if verbose:
        tprint_info(f"🔍 Eigenvalue classification:")
        tprint_info(f"   - Signal eigenvalues: {n_signal}")
        tprint_info(f"   - Noise eigenvalues: {n_noise}")
        tprint_info(f"   - Noise ratio: {n_noise/n_assets:.1%}")
    
    # Apply denoising method
    if method == 'shrink':
        # Shrink noise eigenvalues to average
        if n_noise > 0:
            noise_eigenvalues = eigenvalues[noise_mask]
            signal_eigenvalues = eigenvalues[signal_mask]
            
            # Calculate shrinkage target
            if n_signal > 0:
                shrink_target = np.mean(signal_eigenvalues)
            else:
                shrink_target = np.mean(noise_eigenvalues)
            
            # Apply shrinkage
            eigenvalues_denoised = eigenvalues.copy()
            eigenvalues_denoised[noise_mask] = shrink_target
            
            if verbose:
                tprint_info(f"🔧 Shrinkage method:")
                tprint_info(f"   - Shrink target: {shrink_target:.4f}")
                tprint_info(f"   - Original noise mean: {np.mean(noise_eigenvalues):.4f}")
        else:
            eigenvalues_denoised = eigenvalues.copy()
            if verbose:
                tprint_info("ℹ️ No noise eigenvalues detected, using original")
    
    elif method == 'filter':
        # Filter out noise eigenvalues (set to small positive value)
        eigenvalues_denoised = eigenvalues.copy()
        if n_noise > 0:
            eigenvalues_denoised[noise_mask] = 1e-6  # Small positive value
        
        if verbose:
            tprint_info(f"🔧 Filter method: {n_noise} eigenvalues filtered")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'shrink' or 'filter'")
    
    # Reconstruct correlation matrix
    correlation_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    
    # Ensure diagonal is 1 (correlation matrix property)
    np.fill_diagonal(correlation_denoised, 1.0)
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(
        correlation_denoised,
        index=corr_df.index,
        columns=corr_df.columns
    )
    
    # Validate result
    if not np.allclose(np.diag(result_df.values), 1.0, atol=1e-6):
        tprint_warning("⚠️ Diagonal elements not exactly 1.0, forcing correction")
        np.fill_diagonal(result_df.values, 1.0)
    
    # Check for negative eigenvalues (shouldn't happen with proper denoising)
    final_eigenvalues = np.linalg.eigvalsh(result_df.values)
    if np.any(final_eigenvalues < -1e-8):
        tprint_warning("⚠️ Negative eigenvalues detected, applying correction")
        # Ensure positive semi-definite
        result_df = ensure_positive_semi_definite(result_df)
    
    if verbose:
        # Calculate improvement metrics
        original_condition = np.linalg.cond(corr_df.values)
        denoised_condition = np.linalg.cond(result_df.values)
        
        tprint_success("✅ Marcenko-Pastur denoising complete:")
        tprint_info(f"   - Condition number: {original_condition:.2f} → {denoised_condition:.2f}")
        tprint_info(f"   - Frobenius norm change: {np.linalg.norm(result_df.values - corr_df.values):.4f}")
        tprint_info(f"   - Max absolute change: {np.max(np.abs(result_df.values - corr_df.values)):.4f}")
    
    return result_df

def ensure_positive_semi_definite(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure matrix is positive semi-definite using eigenvalue correction.
    
    Args:
        matrix: Input matrix (DataFrame)
        
    Returns:
        Positive semi-definite matrix (DataFrame)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.values)
    
    # Set negative eigenvalues to small positive value
    eigenvalues_corrected = np.maximum(eigenvalues, 1e-8)
    
    # Reconstruct matrix
    corrected_matrix = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T
    
    result = pd.DataFrame(
        corrected_matrix,
        index=matrix.index,
        columns=matrix.columns
    )
    
    return result

class CovarianceDenoiser:
    """
    Configurable covariance matrix denoiser using Marcenko-Pastur distribution.
    """
    
    def __init__(
        self,
        method: str = 'shrink',
        verbose: bool = True,
        validate_input: bool = True
    ):
        """
        Initialize covariance denoiser.
        
        Args:
            method: Denoising method ('shrink' or 'filter')
            verbose: Whether to print diagnostic information
            validate_input: Whether to validate input matrices
        """
        self.method = method
        self.verbose = verbose
        self.validate_input = validate_input
        self.last_results_ = None
    
    def fit_transform(self, correlation_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Fit denoiser and transform correlation matrix.
        
        Args:
            correlation_matrix: Input correlation matrix
            
        Returns:
            Denoised correlation matrix
        """
        if self.validate_input:
            self._validate_input(correlation_matrix)
        
        result = denoise_correlation_mp(
            correlation_matrix,
            method=self.method,
            verbose=self.verbose
        )
        
        self.last_results_ = {
            'original_matrix': correlation_matrix.copy(),
            'denoised_matrix': result.copy(),
            'method': self.method
        }
        
        return result
    
    def _validate_input(self, matrix: pd.DataFrame) -> None:
        """Validate input correlation matrix."""
        if not isinstance(matrix, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input must be DataFrame or numpy array")
        
        if isinstance(matrix, np.ndarray):
            matrix = pd.DataFrame(matrix)
        
        # Check if square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
        
        # Check diagonal
        if not np.allclose(np.diag(matrix.values), 1.0, atol=1e-3):
            if self.verbose:
                tprint_warning("⚠️ Input diagonal not close to 1.0, normalizing...")
            np.fill_diagonal(matrix.values, 1.0)
        
        # Check symmetry
        if not np.allclose(matrix.values, matrix.values.T, atol=1e-6):
            if self.verbose:
                tprint_warning("⚠️ Input not symmetric, symmetrizing...")
            matrix.values = (matrix.values + matrix.values.T) / 2
    
    def get_last_results(self) -> Optional[dict]:
        """Get results from last denoising operation."""
        return self.last_results_

# Convenience function for quick usage
def quick_denoise(correlation_matrix: pd.DataFrame, method: str = 'shrink') -> pd.DataFrame:
    """
    Quick covariance denoising with default settings.
    
    Args:
        correlation_matrix: Input correlation matrix
        method: Denoising method
        
    Returns:
        Denoised correlation matrix
    """
    denoiser = CovarianceDenoiser(method=method, verbose=False)
    return denoiser.fit_transform(correlation_matrix)
