"""
Fast Causal Proxy
=================

Implements a fast proxy for Causal Discovery using Inverse Covariance (Precision Matrix)
estimation via Graphical Lasso. This approximates the skeleton of the causal graph
significantly faster than PC algorithm (O(M^3) vs O(2^M)).

Uses:
- GraphicalLassoCV for sparse inverse covariance estimation.
- Partial Correlation derivation from precision matrix.
- Thresholding to identify significant edges.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Sklearn imports
try:
    from sklearn.covariance import GraphicalLassoCV, EmpiricalCovariance
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class FastCausalProxy:
    """
    Fast proxy for Causal Discovery using Gaussian Graphical Models (Inverse Covariance).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Sklearn not available. FastCausalProxy will fail.")

    def discover_structure_proxy(
        self,
        df: pd.DataFrame,
        alpha_threshold: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Discover causal graph skeleton using Precision Matrix proxy.

        Args:
            df: Input dataframe (numeric only)
            alpha_threshold: Significance threshold for partial correlations (approx)

        Returns:
            Dictionary representing the graph (adjacency list)
        """
        if df.empty or df.shape[1] < 2:
            return {}

        try:
            if self.verbose:
                tprint_info(f"🚀 Fast Causal Proxy: analyzing {df.shape[1]} variables...")

            # 1. Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df)

            # 2. Estimate Precision Matrix (Inverse Covariance)
            # Use GraphicalLassoCV for automatic regularization selection
            # It enforces sparsity (L1 penalty)
            try:
                model = GraphicalLassoCV(cv=3, n_jobs=-1, assume_centered=True)
                model.fit(X_scaled)
                precision_matrix = model.precision_
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ GraphicalLasso failed ({e}), falling back to EmpiricalCovariance")
                model = EmpiricalCovariance(assume_centered=True)
                model.fit(X_scaled)
                precision_matrix = model.precision_

            # 3. Convert to Partial Correlations
            # rho_ij = -p_ij / sqrt(p_ii * p_jj)
            d = np.sqrt(np.diag(precision_matrix))
            partial_corr = precision_matrix / np.outer(d, d)
            np.fill_diagonal(partial_corr, 1.0) # Diagonal is 1.0

            # Off-diagonal elements should be negated for true partial corr, but we care about magnitude
            # Actual formula: rho_ij = -P_ij / ... for i != j.
            # We use absolute value for edge detection.

            # 4. Construct Graph Skeleton
            # Threshold: simple heuristic or statistical test.
            # GraphicalLasso already enforces sparsity, so non-zero elements are "selected".
            # But we can apply an additional small threshold to remove noise if using EmpiricalCovariance

            adjacency = {col: [] for col in df.columns}
            cols = df.columns
            n_edges = 0

            # Threshold: if GraphicalLasso was used, it's sparse. If Empirical, it's dense.
            # We use a robust threshold: 0.05 or based on sample size 1/sqrt(N)
            robust_threshold = max(alpha_threshold, 2.0 / np.sqrt(len(df)))

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    # Check partial correlation magnitude
                    # Note: We check P_ij. If P_ij is 0, they are conditionally independent.
                    # Partial corr preserves this zero structure.

                    val = partial_corr[i, j]

                    if abs(val) > robust_threshold:
                        # Edge exists (undirected in skeleton)
                        # We add bidirectional for skeleton representation
                        adjacency[cols[i]].append(cols[j])
                        adjacency[cols[j]].append(cols[i])
                        n_edges += 1

            if self.verbose:
                tprint_success(f"✅ Fast Causal Proxy complete: {n_edges} edges found (Skeleton)")

            return adjacency

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Fast Causal Proxy failed: {e}")
            return {}
