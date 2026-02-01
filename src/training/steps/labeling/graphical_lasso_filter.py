"""
Graphical Lasso Filter
======================

Implements a filter for Causal Discovery using Inverse Covariance (Precision Matrix)
estimation via Graphical Lasso (Gaussian Graphical Models). This serves as a
regularized dependency filter to prune edges found by other causal discovery algorithms
(like PC or LiNGAM) that are conditionally independent in the linear Gaussian limit.

AFML Perspective (De Prado):
----------------------------
1.  **Hypothesis Generator/Filter:**
    Graphical Lasso is a **regularized dependency filter**, not evidence of causality.
    It is used here to identify conditionally redundant features and sanity-check
    engineered features.

2.  **Valid Use Cases:**
    -   **Sanity Checking:** If two features are conditionally independent ($P_{ij} \approx 0$),
        enforcing interaction in models is suspect.
    -   **Constraint-Aware Modeling:** Use sparsity to limit interactions.

3.  **Invalid Use Cases (Category Errors):**
    -   Inferring causal direction directly.
    -   Claiming causality without temporal falsification.

Algorithm Details:
------------------
1.  **Graphical Lasso (Glasso):**
    Estimates Precision Matrix via L1-regularized MLE. Induces sparsity.

2.  **Pruning Mechanism:**
    For an edge $A \to B$ discovered by the primary algorithm:
    -   **Glasso Check:** Is there a significant partial correlation $\rho_{AB}$?
    -   **Temporal Check:** Is there a significant lagged correlation ($A_t \to B_{t+1}$ or $B_t \to A_{t+1}$)?

    If both checks fail, the edge is pruned.
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

class GraphicalLassoFilter:
    """
    Filter for Causal Discovery using Gaussian Graphical Models (Inverse Covariance)
    and Temporal Falsification.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Sklearn not available. GraphicalLassoFilter will fail.")

    def filter_graph(
        self,
        causal_graph: Dict[str, List[str]],
        df: pd.DataFrame,
        alpha_threshold: float = 0.05,
        enable_temporal_pruning: bool = True,
        temporal_lag: int = 1
    ) -> Dict[str, List[str]]:
        """
        Prune edges from an existing causal graph using Glasso and Temporal checks.

        Args:
            causal_graph: Input graph {parent: [children]}
            df: Input dataframe (numeric only)
            alpha_threshold: Significance threshold for partial correlations
            enable_temporal_pruning: If True, also check lagged correlations
            temporal_lag: Lag for temporal check

        Returns:
            Pruned causal graph
        """
        if df.empty or not causal_graph:
            return causal_graph

        try:
            if self.verbose:
                tprint_info(f"🛡️ Graphical Lasso Filter: Validating {sum(len(v) for v in causal_graph.values())} edges...")

            # 1. Standardize using RobustScaler for financial data
            from sklearn.preprocessing import RobustScaler
            
            # Additional robustness: drop duplicates if any (collinear features break Glasso)
            df_dedup = df.T.drop_duplicates().T
            if df_dedup.shape[1] < 2:
                return causal_graph
                
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(df_dedup)
            
            # Clip extreme values to prevent numerical explosion
            X_scaled = np.clip(X_scaled, -10, 10)
            
            col_map = {name: i for i, name in enumerate(df_dedup.columns)}

            # 2. Estimate Precision Matrix (Glasso)
            try:
                # Max iter increased for convergence, with error suppression
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = GraphicalLassoCV(
                        cv=3,
                        n_jobs=1,
                        assume_centered=True,
                        max_iter=2000,
                        tol=1e-3
                    )
                    model.fit(X_scaled)
                precision_matrix = model.precision_
                
                # Validation
                if not np.isfinite(precision_matrix).all() or np.isnan(precision_matrix).any():
                    raise RuntimeError("Non-finite precision matrix")
                    
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ GraphicalLasso failed/diverged ({e}), falling back to EmpiricalCovariance")
                model = EmpiricalCovariance(assume_centered=True)
                model.fit(X_scaled)
                precision_matrix = model.precision_

            # 3. Convert to Partial Correlations
            d = np.sqrt(np.diag(precision_matrix))
            partial_corr = precision_matrix / np.outer(d, d)

            # Threshold
            robust_threshold = max(alpha_threshold, 2.0 / np.sqrt(len(df)))

            # 4. Prune Edges
            pruned_graph = {k: [] for k in causal_graph.keys()}
            edges_removed_glasso = 0
            edges_removed_temporal = 0

            for parent, children in causal_graph.items():
                if parent not in col_map:
                    continue
                idx_p = col_map[parent]

                for child in children:
                    if child not in col_map:
                        continue
                    idx_c = col_map[child]

                    keep_edge = True

                    # A) Glasso Check (Conditional Independence)
                    # If partial correlation is near zero, edges are likely spurious
                    p_corr = abs(partial_corr[idx_p, idx_c])
                    if p_corr < robust_threshold:
                        edges_removed_glasso += 1
                        keep_edge = False

                    # B) Temporal Check (Falsification)
                    # If Glasso passed, we double check with temporal lag if enabled
                    if keep_edge and enable_temporal_pruning:
                        temporal_link = self._check_temporal_link(df, parent, child, temporal_lag, robust_threshold)
                        if not temporal_link:
                            edges_removed_temporal += 1
                            keep_edge = False

                    if keep_edge:
                        pruned_graph[parent].append(child)

            # Cleanup empty keys if desired, but keeping structure is safer

            if self.verbose:
                total_removed = edges_removed_glasso + edges_removed_temporal
                tprint_success(f"✅ Graph Filtering Complete: Removed {total_removed} edges")
                tprint_info(f"   - Glasso Pruning (Cond. Indep.): {edges_removed_glasso}")
                tprint_info(f"   - Temporal Pruning (Falsification): {edges_removed_temporal}")

            return pruned_graph

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Graphical Lasso Filter failed: {e}")
            return causal_graph

    def fit_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate the connectivity of each feature using Graphical Lasso.
        Returns a dictionary of {feature: connectivity_score}.
        Score is the sum of absolute partial correlations with other features.
        """
        if df.empty:
            return {}

        try:
            if self.verbose:
                tprint_info(f"🛡️ Graphical Lasso Scoring: Analyzing {len(df.columns)} features...")

            # 1. Standardize
            from sklearn.preprocessing import RobustScaler
            
            # Drop duplicates to prevent singularity
            df_dedup = df.T.drop_duplicates().T
            if df_dedup.shape[1] < 2:
                return {c: 0.0 for c in df.columns}
                
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(df_dedup)
            X_scaled = np.clip(X_scaled, -10, 10)
            
            # 2. Estimate Precision Matrix
            try:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    model = GraphicalLassoCV(cv=3, n_jobs=1, assume_centered=True, max_iter=2000, tol=1e-3)
                    model.fit(X_scaled)
                precision_matrix = model.precision_
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ GraphicalLasso failed ({e}), using EmpiricalCovariance")
                model = EmpiricalCovariance(assume_centered=True)
                model.fit(X_scaled)
                precision_matrix = model.precision_

            # 3. Partial Correlations
            d = np.sqrt(np.diag(precision_matrix))
            partial_corr = precision_matrix / np.outer(d, d)
            
            # 4. Compute Scores (Sum of absolute off-diagonal partial correlations)
            scores = {}
            for i, col in enumerate(df_dedup.columns):
                # Sum of abs correlations with others
                p_vec = np.abs(partial_corr[i, :])
                p_vec[i] = 0.0 # Zero out self
                scores[col] = float(np.sum(p_vec))
                
            return scores

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Graphical Lasso Scoing failed: {e}")
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
        """
        series_a = df[col_a]
        series_b = df[col_b]

        # A causing B: A(t-lag) -> B(t)
        # Using correlation of A shifted forward vs B? No.
        # A(t) and B(t+1).
        # Shift A by +1 (forward) -> No, shift operations usually mean:
        # shift(1) moves t to t+1 (data moves down).
        # We want corr(A[t], B[t+1]).
        # Series A: [a0, a1, a2...]
        # Series B: [b0, b1, b2...]
        # We want to align a0 with b1.
        # df[col_a] vs df[col_b].shift(-1)?
        # df.shift(1): [NaN, b0, b1...] (t aligned with t-1)

        # Check: A(t) -> B(t+1)
        # Corr(A, B_shifted_back_to_align)
        # We align A[t] with B[t+1].
        # B.shift(-1) puts B[t+1] at index t.
        corr_a_b = series_a.corr(series_b.shift(-lag))

        # B causing A: B(t) -> A(t+1)
        corr_b_a = series_b.corr(series_a.shift(-lag))

        # Relaxed threshold for temporal check (noise is higher)
        lag_threshold = threshold * 0.8

        if abs(corr_a_b) > lag_threshold or abs(corr_b_a) > lag_threshold:
            return True

        return False
