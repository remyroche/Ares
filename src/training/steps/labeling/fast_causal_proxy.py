"""
Fast Causal Proxy
=================

Implements a fast proxy for Causal Discovery using Inverse Covariance (Precision Matrix)
estimation via Graphical Lasso (Gaussian Graphical Models). This approximates the
skeleton of the causal graph (undirected edges) significantly faster than the
standard PC (Peter-Clark) algorithm.

AFML Perspective (De Prado):
----------------------------
1.  **Hypothesis Generator, Not Causal Proof:**
    In Financial Machine Learning, causality cannot be inferred from correlations,
    partial or otherwise, without temporal structure and falsification.
    Graphical Lasso is a **regularized dependency filter**, not evidence of causality.

2.  **Valid Use Cases:**
    -   **Feature Interaction Pruning:** Identify conditionally redundant features
        to reduce feature graph density.
    -   **Sanity Checking:** If two features are conditionally independent ($P_{ij} \approx 0$),
        enforcing interaction in models is suspect.
    -   **Constraint-Aware Modeling:** Use sparsity to limit interactions or guide
        monotonic/additive structures.

3.  **Invalid Use Cases (Category Errors):**
    -   Inferring causal direction (Glasso returns undirected edges).
    -   Justifying economic narratives based solely on the graph.
    -   Choosing trading signals directly.
    -   Claiming "Feature A causes Returns".

4.  **Temporal Falsification:**
    This implementation includes a **Time-Lagged Partial Correlation Filter**.
    Edges found by Graphical Lasso (A-B) are pruned if they fail to show predictive
    power in a temporal setting (e.g., A(t) -> B(t+1) or B(t) -> A(t+1)).
    This helps filter out purely contemporaneous correlations that lack predictive utility.

Algorithm Details:
------------------
1.  **Gaussian Graphical Models (GGM):**
    Assumes multivariate Gaussian distribution. Zeros in Precision Matrix $\Sigma^{-1}$
    correspond to conditional independence ($X_i \perp X_j | V \setminus \{X_i, X_j\}$).

2.  **Graphical Lasso (Glasso):**
    Estimates Precision Matrix via L1-regularized MLE. Induces sparsity (neighborhood selection).

3.  **Partial Correlation:**
    $\rho_{ij} = -\frac{\omega_{ij}}{\sqrt{\omega_{ii} \omega_{jj}}}$. Non-zero values imply an edge.

4.  **Temporal Pruning (New):**
    For each edge $(i, j)$ in the skeleton, we verify if $|Corr(X_i(t), X_j(t+1))| > \epsilon$
    or $|Corr(X_j(t), X_i(t+1))| > \epsilon$. If neither direction holds, the edge is likely
    a contemporaneous artifact or confounder effect without predictive value, and is pruned.
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
    Fast proxy for Causal Discovery using Gaussian Graphical Models (Inverse Covariance)
    with temporal falsification.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Sklearn not available. FastCausalProxy will fail.")

    def discover_structure_proxy(
        self,
        df: pd.DataFrame,
        alpha_threshold: float = 0.05,
        prune_non_temporal: bool = True,
        temporal_lag: int = 1
    ) -> Dict[str, List[str]]:
        """
        Discover causal graph skeleton using Precision Matrix proxy with optional temporal pruning.

        Args:
            df: Input dataframe (numeric only)
            alpha_threshold: Significance threshold for partial correlations (approx)
            prune_non_temporal: If True, remove edges that lack lagged correlation support.
            temporal_lag: Lag to check for temporal structure (default 1).

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
                # Max iter increased for convergence on complex datasets
                model = GraphicalLassoCV(cv=3, n_jobs=-1, assume_centered=True, max_iter=500)
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

            # 4. Construct Graph Skeleton
            adjacency = {col: [] for col in df.columns}
            cols = df.columns
            n_edges = 0
            n_pruned_temporal = 0

            # Threshold robustness
            robust_threshold = max(alpha_threshold, 2.0 / np.sqrt(len(df)))

            # Pre-compute lagged correlations if pruning is enabled
            if prune_non_temporal:
                # Create lagged DataFrame once
                df_lagged = df.shift(temporal_lag)
                # We need correlations between X(t) and X(t-1)
                # Compute correlation matrix of concatenated [df, df_lagged] is efficient but messy to index
                # Iterative check is fine for sparse edges found by Glasso
                pass

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = partial_corr[i, j]

                    if abs(val) > robust_threshold:
                        # Candidate Edge found by Glasso
                        keep_edge = True

                        if prune_non_temporal:
                            keep_edge = self._check_temporal_link(df, cols[i], cols[j], temporal_lag, robust_threshold)
                            if not keep_edge:
                                n_pruned_temporal += 1

                        if keep_edge:
                            adjacency[cols[i]].append(cols[j])
                            adjacency[cols[j]].append(cols[i])
                            n_edges += 1

            if self.verbose:
                tprint_success(f"✅ Fast Causal Proxy complete: {n_edges} edges found (Skeleton)")
                if prune_non_temporal:
                    tprint_info(f"   ✂️ Temporal Pruning: Removed {n_pruned_temporal} edges lacking time-lagged support")

            return adjacency

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Fast Causal Proxy failed: {e}")
            return {}

    def _check_temporal_link(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        lag: int,
        threshold: float
    ) -> bool:
        """
        Check if there is a significant lagged correlation in either direction.
        A(t) -> B(t+lag)  OR  B(t) -> A(t+lag)

        If neither is significant, the edge (A-B) is likely purely contemporaneous
        and should be pruned for causal modeling purposes.
        """
        # Calculate lagged correlations
        # We need corr(A_t, B_{t+k}) and corr(B_t, A_{t+k})

        # A leads B? (A_t, B_{t+1})
        # Correlation between A and B shifted backwards (B_{t+1} aligned with A_t)
        # Actually: df[col_a] vs df[col_b].shift(-lag)
        # Or easier: df[col_a].shift(lag) vs df[col_b] (A_{t-1} vs B_t)

        series_a = df[col_a]
        series_b = df[col_b]

        # A causing B: A(t-lag) -> B(t)
        corr_a_b = series_a.shift(lag).corr(series_b)

        # B causing A: B(t-lag) -> A(t)
        corr_b_a = series_b.shift(lag).corr(series_a)

        # Check if either direction is significant (using slightly relaxed threshold for lag check)
        # We use 0.5 * threshold to be permissive but still filter zero-lag noise
        lag_threshold = threshold * 0.8

        if abs(corr_a_b) > lag_threshold or abs(corr_b_a) > lag_threshold:
            return True

        return False
