"""
Causal Discovery Light Module - NOTEARS Implementation

Implements a lightweight, fast causal discovery using Linear NOTEARS optimization
instead of combinatorial PC algorithm. Designed for high performance and memory efficiency.

Key Features:
1. Linear NOTEARS (Non-parametric Optimization for TEaring AR Structures)
2. Scipy L-BFGS-B optimization with analytical gradients
3. Robust feature scaling
4. Memory efficient (float32, in-place)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler, RobustScaler
from numba import njit
import sys
import gc

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

@njit(fastmath=True, cache=True)
def _numba_loss_l2(W, X):
    """
    Numba optimized L2 loss and gradient.
    L = 0.5/n * ||X - XW||^2
    """
    n = X.shape[0]
    # Ensure dtype compatibility for matrix multiplication
    W_float = W.astype(np.float64)
    X_float = X.astype(np.float64)
    M = X_float @ W_float
    R = X_float - M
    loss = 0.5 / n * np.sum(R ** 2)
    G_loss = - 1.0 / n * (X_float.T @ R)
    return loss, G_loss

@njit(fastmath=True, cache=True)
def _numba_loss_logistic(W, X):
    """
    Numba optimized Logistic loss and gradient.
    """
    n = X.shape[0]
    # Ensure dtype compatibility for matrix multiplication
    W_float = W.astype(np.float64)
    X_float = X.astype(np.float64)
    M = X_float @ W_float
    # logaddexp(0, M)
    loss = 1.0 / n * np.sum(np.logaddexp(0.0, M) - X_float * M)
    # sigmoid(M)
    sig_M = 1.0 / (1.0 + np.exp(-M))
    G_loss = 1.0 / n * (X_float.T @ (sig_M - X_float))
    return loss, G_loss

# _h remains non-JIT due to scipy.expm dependency (matrix exponential)
# unless we implement Pade approximation in Numba, which is overkill for "light" request.

class CausalDiscoveryLight:
    """
    Lightweight Causal Discovery using Linear NOTEARS.
    Optimized with Numba JIT for loss calculation.
    """
    # ... __init__ kept as is ...
    def __init__(
        self,
        significance_level: float = 0.05,
        lambda1: float = 0.1,
        loss_type: str = 'l2',
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e+16,
        w_threshold: float = 0.3, # Threshold for edge filtering
        verbose: bool = True
    ):
        """
        Initialize Light Causal Discovery.
        """
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.verbose = verbose
        
        # Compatibility
        self.significance_level = significance_level
        self.causal_graph_ = None
        self.adjacency_matrix_ = None
        
        if self.verbose:
            tprint_info("⚡ CausalDiscoveryLight: Initializing NOTEARS engine (Numba Accelerated)...")
            tprint_info(f"   ⚙️ Params: lambda1={lambda1}, thresh={w_threshold}, max_iter={max_iter}")

    def _loss(self, W, X):
        """Evaluate value and gradient of loss - Dispatches to Numba."""
        if self.loss_type == 'l2':
            return _numba_loss_l2(W, X)
        elif self.loss_type == 'logistic':
            return _numba_loss_logistic(W, X)
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint."""
        # h(W) = Tr(exp(W*W)) - d
        d = W.shape[0]
        # P = exp(W*W), G_h = P.T * 2W
        # Using scipy.linalg.expm for stability
        # Cast to float64 for matrix exp stability then back?
        # Scipy handles it.
        P = slin.expm(W * W) 
        h = np.trace(P) - d
        G_h = P.T @ (W * 2)
        return h, G_h

    # ... _adj, _func, linear_notears kept mostly similar ...
    
    def _adj(self, w):
        """Convert doubled variables (w) back to adjacency matrix (W)."""
        d = int(np.sqrt(w.shape[0] // 2))
        return (w[:d*d] - w[d*d:]).reshape([d, d])

    def _func(self, w, X, rho, alpha):
        """Augmented Lagrangian function + L1 (smooth approximation)."""
        d = X.shape[1]
        W = self._adj(w)
        loss, G_loss = self._loss(W, X)
        h, G_h = self._h(W)
        
        obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        
        g_obj = np.concatenate((G_smooth.flatten() + self.lambda1, -G_smooth.flatten() + self.lambda1))
        return obj, g_obj

    def linear_notears(self, X: np.ndarray) -> np.ndarray:
        # ... logic same as before ...
        n, d = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        
        if self.verbose:
             tprint_info(f"   🚀 NOTEARS: Opt start. n={n}, d={d}")
             
        for i in range(self.max_iter):
            w_new, h_new = None, None
            while rho < self.rho_max:
                res = sopt.minimize(
                    self._func, w_est, args=(X, rho, alpha), method='L-BFGS-B', jac=True, bounds=bnds
                )
                w_new = res.x
                h_new, _ = self._h(self._adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= self.h_tol or rho >= self.rho_max:
                break
                
        W_est = self._adj(w_est)
        W_est[np.abs(W_est) < self.w_threshold] = 0
        return W_est

    def run_discovery(self, data: pd.DataFrame, variable_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Main execution method (Compatible with GMM usage).
        """
        try:
            if self.verbose:
                tprint_info("🔄 Light Discovery: Starting...")
                tprint_info(f"   📊 Max Memory Check: {sys.getsizeof(data)/1e6:.2f} MB")
            
            # 1. Enforce Float32 and Memory Cleanup
            X_raw = data.values.astype(np.float32)
            gc.collect() # aggressive gc
            
            # Subsampling safeguard
            if X_raw.shape[0] > 10000:
                tprint_warning("   ⚠️ Data too large for light discovery, subsampling to 10000.")
                indices = np.linspace(0, X_raw.shape[0]-1, 10000).astype(int)
                X_raw = X_raw[indices]
            
            # Remove constant columns
            std = np.std(X_raw, axis=0)
            valid_cols = std > 1e-9
            if not np.all(valid_cols):
                 tprint_warning(f"   ⚠️ Dropping {np.sum(~valid_cols)} constant columns")
                 X_raw = X_raw[:, valid_cols]
                 if variable_names:
                     variable_names = [n for i, n in enumerate(variable_names) if valid_cols[i]]
            
            tprint_info("   ⚖️ Scaling features (Robust + Standard)...")
            X_scaled = RobustScaler().fit_transform(X_raw)
            X_scaled = StandardScaler().fit_transform(X_scaled)
            
            # Ensure float32 after scaling (sklearn outputs float64)
            X_scaled = X_scaled.astype(np.float32)

            tprint_info("   🏃 Running Linear NOTEARS optimization (JIT)...")
            W_est = self.linear_notears(X_scaled)
            
            if variable_names is None:
                variable_names = list(data.columns) if hasattr(data, 'columns') else [f"X{i}" for i in range(data.shape[1])]
            
            graph = {var: [] for var in variable_names}
            n_edges = 0
            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if W_est[i, j] != 0:
                        graph[variable_names[i]].append(variable_names[j])
                        n_edges += 1
            
            self.adjacency_matrix_ = W_est
            self.causal_graph_ = graph
            
            tprint_success(f"✅ Light Discovery Complete: Found {n_edges} edges.")
            return graph
            
        except Exception as e:
            tprint_error(f"❌ Light Discovery Failed: {e}")
            import traceback
            tprint_error(traceback.format_exc())
            return {}


    # Alias for compatibility if needed
    def pc_algorithm(self, data: pd.DataFrame, variable_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Alias for run_discovery to maintain API compatibility."""
        return self.run_discovery(data, variable_names)
