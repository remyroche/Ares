"""
Statistical utilities for De Prado Feature Engine.
Includes Marchenko-Pastur denoising and advanced correlation metrics.
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.neighbors import KernelDensity

def get_pca(matrix):
    """
    Get eigenvalues and eigenvectors of a Hermitian matrix.
    """
    e_val, e_vec = np.linalg.eigh(matrix)
    indices = e_val.argsort()[::-1]
    e_val, e_vec = e_val[indices], e_vec[:, indices]
    e_val = np.diagflat(e_val)
    return e_val, e_vec

def fit_kde(obs, b_width=0.25, kernel='gaussian', x=None):
    """
    Fit kernel to a series of obs, and derive the prob of obs
    x is the array of values on which the fit KDE will be evaluated
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf

def get_marchenko_pastur_pdf(var, q, pts):
    """
    Generate Marchenko-Pastur PDF.
    """
    e_min = var * (1 - (1./q)**.5)**2
    e_max = var * (1 + (1./q)**.5)**2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q/(2*np.pi*var*e_val) * ((e_max-e_val)*(e_val-e_min))**.5
    pdf = pd.Series(pdf.flatten(), index=e_val.flatten())
    return pdf

def find_max_eval(e_val, q, b_width):
    """
    Find max random eVal by fitting Marchenko's dist to the empirical one.
    """
    out = {'max_eval': None, 'var': None}
    e_val = np.diag(e_val).copy()
    e_val = e_val[e_val > 1e-9] # Ignore near-zeros
    
    # Grid search for variance
    from scipy.optimize import minimize
    
    def err_pdfs(var, e_val, q, b_width, pts=1000):
        pdf0 = get_marchenko_pastur_pdf(var[0], q, pts) # Theoretical
        pdf1 = fit_kde(e_val, b_width, x=pdf0.index.values) # Empirical
        sse = np.sum((pdf1 - pdf0)**2)
        return sse

    if len(e_val) == 0:
        return 0, 1.0

    x0 = np.array([0.5])
    # Bounds for variance
    bounds = ((1e-5, 1-1e-5),)
    
    try:
        res = minimize(err_pdfs, x0, args=(e_val, q, b_width), bounds=bounds)
        var = res.x[0]
        e_max = var * (1 + (1./q)**.5)**2
        return e_max, var
    except:
        # Fallback if optimization fails
        return np.max(e_val), 1.0

def denoise_covariance(cov, q, b_width=0.01):
    """
    Remove noise from covariance matrix using Marchenko-Pastur theorem.
    Standard 'Constant Residual Eigenvalue' method.
    
    Args:
        cov: Covariance matrix (pandas DataFrame or numpy array)
        q: T/N ratio (rows/cols)
        b_width: Bandwidth for KDE
        
    Returns:
        Denoised covariance matrix
    """
    # 1. Get eigenvalues and eigenvectors
    if isinstance(cov, pd.DataFrame):
        cov_val = cov.values
    else:
        cov_val = cov
        
    # Convert to correlation for spectral analysis
    vols = np.sqrt(np.diag(cov_val))
    corr = cov_val / np.outer(vols, vols)
    corr[np.isnan(corr)] = 0.0
    
    e_val0, e_vec0 = get_pca(corr)
    
    # 2. Find max random eigenvalue
    e_max0, var0 = find_max_eval(e_val0, q, b_width)
    
    # 3. Remove noise
    n_facts = e_val0.shape[0]
    e_val1 = np.diag(e_val0).copy()
    
    # Identify signals vs noise
    # Condition: Eigenvalue > Max Theoretical Random Eigenvalue
    n_signals = np.sum(e_val1 > e_max0)
    
    if n_signals < n_facts:
        # Average the noise eigenvalues
        e_val1[n_signals:] = e_val1[n_signals:].mean()
        
    # Reconstruct correlation matrix
    e_val1 = np.diag(e_val1)
    # cov = V * Lambda * V.T
    corr1 = np.dot(e_vec0, np.dot(e_val1, e_vec0.T))
    
    # Rescale back to covariance
    # Ensure diagonal is 1 for correlation
    np.fill_diagonal(corr1, 1.0)
    cov1 = corr1 * np.outer(vols, vols)
    
    if isinstance(cov, pd.DataFrame):
        return pd.DataFrame(cov1, index=cov.index, columns=cov.columns)
    return cov1

def denoise_correlation(corr, q, b_width=0.01):
    """
    Denoise a correlation matrix directly.
    """
    if isinstance(corr, pd.DataFrame):
        vals = corr.values
    else:
        vals = corr
        
    e_val0, e_vec0 = get_pca(vals)
    e_max0, var0 = find_max_eval(e_val0, q, b_width)
    
    n_facts = e_val0.shape[0]
    e_val1 = np.diag(e_val0).copy()
    n_signals = np.sum(e_val1 > e_max0)
    
    if n_signals < n_facts:
        e_val1[n_signals:] = e_val1[n_signals:].mean()
        
    e_val1 = np.diag(e_val1)
    corr1 = np.dot(e_vec0, np.dot(e_val1, e_vec0.T))
    np.fill_diagonal(corr1, 1.0)
    
    if isinstance(corr, pd.DataFrame):
        return pd.DataFrame(corr1, index=corr.index, columns=corr.columns)
    return corr1

@njit(fastmath=True)
def get_partial_corr_pairwise(corr_matrix, x_idx, y_idx):
    """
    Calculate partial correlation between features x and y, 
    controlling for all other features in the matrix.
    Uses the inverse correlation matrix (Precision Matrix).
    
    Note: Requires inverting the matrix first.
    """
    # This is just a placeholder. True partial correlation requires inverse.
    # The caller should invert the matrix once.
    pass

def get_precision_matrix(corr):
    """
    Get Precision Matrix (Inverse of Correlation Matrix).
    Uses Moore-Penrose pseudo-inverse for stability.
    """
    if isinstance(corr, pd.DataFrame):
        vals = corr.values
    else:
        vals = corr
    return np.linalg.pinv(vals)

@njit(fastmath=True)
def partial_corr_from_precision(prec, i, j):
    """
    Calculate Partial Correlation from Precision Matrix elements.
    rho_xy.z = - P_xy / sqrt(P_xx * P_yy)
    """
    p_ii = prec[i, i]
    p_jj = prec[j, j]
    p_ij = prec[i, j]
    
    if p_ii == 0 or p_jj == 0:
        return 0.0
        
    return -p_ij / np.sqrt(p_ii * p_jj)
