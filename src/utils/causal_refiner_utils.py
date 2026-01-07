"""
Causal Refiner Utilities - De Prado 2026 Protocol
------------------------------------------------
Implementation of Structural Anchors, Marchenko-Pastur Filtering, 
and Spectral Stability for Layer 3 Meta-Models.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Optional, Any

def marchenko_pastur_pdf(var: float, q: float, pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Marchenko-Pastur probability density function.
    q = T / N (Observations / Features)
    """
    e_min = var * (1 - (1./q)**0.5)**2
    e_max = var * (1 + (1./q)**0.5)**2
    e_range = np.linspace(e_min, e_max, pts)
    
    pdf = q / (2 * np.pi * var * e_range) * ((e_max - e_range) * (e_range - e_min))**0.5
    if isinstance(pdf, np.ndarray):
        pdf[pdf < 0] = 0
    return e_range, pdf

def find_max_eigenvalue(eigenvalues: np.ndarray, q: float, pts: int = 1000) -> float:
    """
    Finds λ_max (the Marchenko-Pastur cutoff) by fitting a KDE to the 
    observed eigenvalues and finding where the noise distribution ends.
    """
    def err_func(var, eigenvalues, q):
        e_range, pdf_mp = marchenko_pastur_pdf(var, q, pts)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(eigenvalues.reshape(-1, 1))
        pdf_kde = np.exp(kde.score_samples(e_range.reshape(-1, 1)))
        sse = np.sum((pdf_kde - pdf_mp)**2)
        return sse

    res = minimize(err_func, x0=np.array([1.0]), args=(eigenvalues, q), bounds=[(1e-5, None)])
    var_fit = res.x[0]
    e_max = var_fit * (1 + (1./q)**0.5)**2
    return e_max

def denoise_covariance(cov: np.ndarray, q: float) -> np.ndarray:
    """
    Replaces noise eigenvalues (λ < λ_max) with their average value.
    """
    e_val, e_vec = np.linalg.eigh(cov)
    indices = np.argsort(e_val)[::-1]
    e_val, e_vec = e_val[indices], e_vec[:, indices]
    
    e_max = find_max_eigenvalue(e_val, q)
    n_facts = len(e_val[e_val > e_max])
    
    e_val_denoised = e_val.copy()
    e_val_denoised[n_facts:] = e_val[n_facts:].mean()
    
    cov_denoised = np.dot(e_vec, e_val_denoised.reshape(-1, 1) * e_vec.T)
    # Ensure correct diagonal
    diag = np.diag(cov)
    diag_denoised = np.diag(cov_denoised)
    cov_denoised *= np.sqrt(diag / diag_denoised).reshape(-1, 1) * np.sqrt(diag / diag_denoised)
    return cov_denoised

def get_spectral_stability(current_loadings: np.ndarray, base_loadings: np.ndarray) -> float:
    """
    Calculates the stability of PC loadings (Cosine Similarity).
    """
    # Align shapes if necessary or ensure they are comparable
    c = current_loadings.flatten()
    b = base_loadings.flatten()
    if len(c) != len(b):
        return 0.0
    
    norm_c = np.linalg.norm(c)
    norm_b = np.linalg.norm(b)
    if norm_c == 0 or norm_b == 0:
        return 1.0
        
    dot = np.dot(c, b)
    similarity = dot / (norm_c * norm_b)
    return float(similarity)

def cluster_specialists_by_correlation(X: pd.DataFrame, max_clusters: int = 5) -> Dict[int, List[str]]:
    """
    Groups specialists using correlation-based clustering to define 'Structural Anchors'.
    """
    from sklearn.cluster import AgglomerativeClustering
    corr = X.corr().fillna(0).values
    # Distance = 1 - Correlation
    dist = 1 - np.abs(corr)
    
    n_clusters = min(max_clusters, X.shape[1])
    if n_clusters < 2:
        return {0: X.columns.tolist()}
        
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
    labels = model.fit_predict(dist)
    
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X.columns[i])
        
    return clusters
