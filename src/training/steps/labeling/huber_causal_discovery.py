"""
Node-Wise Huber Causal Discovery
================================

Replaces the PC algorithm with a robust, time-first, node-wise sparse regression approach.
Uses Huber Regressor to detect edges in a structural causal skeleton.

Principles:
1.  **Time-first design (non-negotiable):** Uses lag-only inputs ($X_{t-k} \to Y_t$).
    Contemporaneous links are not directly modeled to avoid simultaneity bias without
    structural equation modeling.
2.  **Split-first fitting:** Stability is part of detection. We fit on multiple
    time splits and aggregate results.
3.  **Multi-target:** Detects predictive ($X_{t-k} \to Y_t$) and structural
    ($X_{t-k}^i \to X_t^j$) skeletons.
4.  **Huber as a Detector:** Uses Huber loss to handle outliers robustly. Edges are
    detected based on coefficient stability, not just magnitude.
5.  **Persistence:** Saves discovery artifacts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings
from collections import defaultdict
import os
import json
from datetime import datetime

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class HuberCausalDiscovery:
    """
    Node-wise sparse regression using Huber Regressor for Causal Skeleton Discovery.
    """

    def __init__(self,
                 max_lag: int = 1,
                 n_splits: int = 5,
                 epsilon: float = 1.35,
                 alpha: float = 0.0001,
                 stability_threshold: float = 0.4,
                 detection_threshold: float = 0.01,
                 n_jobs: int = -1,
                 verbose: bool = True):
        """
        Initialize Huber Causal Discovery.

        Args:
            max_lag: Maximum lag for input features.
            n_splits: Number of time series splits for stability.
            epsilon: Huber epsilon (robustness parameter).
            alpha: L2 regularization strength.
            stability_threshold: Fraction of splits an edge must appear in (0.0 to 1.0).
            detection_threshold: Minimum coefficient magnitude to count as an edge.
            n_jobs: Parallel jobs.
            verbose: Logging verbosity.
        """
        self.max_lag = max_lag
        self.n_splits = n_splits
        self.epsilon = epsilon
        self.alpha = alpha
        self.stability_threshold = stability_threshold
        self.detection_threshold = detection_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Artifact storage
        self.discovery_artifacts = {}

    def _fit_target_split(self, X_train: np.ndarray, y_train: np.ndarray,
                          predictor_names: List[str]) -> Dict[str, float]:
        """Fit Huber for a single target on a single split."""
        if len(y_train) < 50:
            return {} # Too few samples

        try:
            # Fit Huber
            # Note: Data should be standardized before this call
            model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, fit_intercept=True, max_iter=200)
            model.fit(X_train, y_train)

            # Extract significant coefs
            coeffs = {}
            for name, coef in zip(predictor_names, model.coef_):
                if abs(coef) > self.detection_threshold:
                    coeffs[name] = coef
            return coeffs
        except Exception:
            return {}

    def _process_target_variable(self, target_col: str, X_lagged: np.ndarray,
                                 y_target: np.ndarray, predictor_names: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Process a single target variable across all splits."""

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        split_results = []

        # Execute splits
        # We can parallelize splits, but since we parallelize targets, we do splits sequentially here
        # to avoid oversubscription if n_jobs is global.
        # Actually, sklearn models release GIL, but overhead matters.
        # Let's run splits sequentially inside this function.

        for train_index, _ in tscv.split(X_lagged):
            X_train_split = X_lagged[train_index]
            y_train_split = y_target[train_index]

            coeffs = self._fit_target_split(X_train_split, y_train_split, predictor_names)
            split_results.append(coeffs)

        # Aggregation / Stability Selection
        # Count occurrences of each edge
        edge_counts = defaultdict(int)
        edge_magnitudes = defaultdict(list)

        for res in split_results:
            for pred, coef in res.items():
                edge_counts[pred] += 1
                edge_magnitudes[pred].append(abs(coef))

        # Filter by stability
        stable_parents = []
        parent_strengths = {}

        threshold_count = int(self.n_splits * self.stability_threshold)

        for pred, count in edge_counts.items():
            if count >= threshold_count:
                stable_parents.append(pred)
                parent_strengths[pred] = np.mean(edge_magnitudes[pred])

        return target_col, {
            'parents': stable_parents,
            'strengths': parent_strengths,
            'stability_counts': dict(edge_counts)
        }

    def discover_causal_structure(self, df: pd.DataFrame, target_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the discovery pipeline.

        Args:
            df: Input dataframe (numeric).
            target_variable: (Optional) Focus variable, though we discover full skeleton.

        Returns:
            Dictionary with 'causal_graph' (adjacency list) and metadata.
        """
        if df.empty:
            tprint_error("HuberCausalDiscovery: Empty dataframe.")
            return {}

        if self.verbose:
            tprint_info(f"🚀 Huber Causal Discovery: {df.shape[1]} variables, {len(df)} samples")
            tprint_info(f"   ⚙️ Config: lag={self.max_lag}, splits={self.n_splits}, epsilon={self.epsilon}")

        # 1. Data Preparation
        # Standardize inputs
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

        # Create Lagged Features
        # For each variable X, create X_lag1, X_lag2...
        # We predict Y_t using X_{t-k}

        lagged_data = {}
        target_data = df_scaled.iloc[self.max_lag:].copy() # Align targets (drop first k rows)

        predictor_names = []

        # Build lagged matrix
        # Optimize: Shift whole DF
        for lag in range(1, self.max_lag + 1):
            df_shifted = df_scaled.shift(lag).iloc[self.max_lag:]
            for col in df_shifted.columns:
                name = f"{col}_L{lag}"
                lagged_data[name] = df_shifted[col].values
                predictor_names.append(name)

        X_matrix = pd.DataFrame(lagged_data).values # (N_samples, N_vars * max_lag)

        if self.verbose:
            tprint_info(f"   📊 Feature Matrix: {X_matrix.shape} (lags construction)")

        # 2. Parallel Node-wise Regression
        # We predict *every* column in df (at time t) using X_matrix (time t-k)

        targets = df.columns.tolist()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_target_variable)(
                target, X_matrix, target_data[target].values, predictor_names
            ) for target in targets
        )

        # 3. Construct Graph
        causal_graph = {}
        causal_strength = {}
        stability_stats = {}

        # Map back lagged names to variables
        # "FeatureA_L1" -> "FeatureA" is parent of Target

        total_edges = 0

        for target, res in results:
            parents_clean = set()
            strengths_clean = {}

            for pred_name in res['parents']:
                # Parse original variable name
                # Format: {col}_L{lag}
                # Find last occurrence of _L
                if "_L" in pred_name:
                    parent_var = pred_name.rsplit("_L", 1)[0]
                    # Edge: parent_var -> target
                    parents_clean.add(parent_var)

                    # Aggregate strength (max or mean across lags if multi-lag)
                    s = res['strengths'][pred_name]
                    if parent_var in strengths_clean:
                        strengths_clean[parent_var] = max(strengths_clean[parent_var], s)
                    else:
                        strengths_clean[parent_var] = s

            causal_graph[target] = list(parents_clean)
            # Store strength tuples (parent, strength)
            causal_strength[target] = strengths_clean
            stability_stats[target] = res['stability_counts']
            total_edges += len(parents_clean)

        # 4. Persistence
        self.save_checkpoints(causal_graph, stability_stats)

        if self.verbose:
            tprint_success(f"✅ Huber Discovery Complete: {total_edges} edges found.")
            # Show top node degree
            degrees = {k: len(v) for k, v in causal_graph.items()}
            if degrees:
                max_deg = max(degrees.values())
                max_node = max(degrees, key=degrees.get)
                tprint_info(f"   ℹ️ Max In-Degree: {max_node} ({max_deg} parents)")

        return {
            'causal_graph': causal_graph,  # {child: [parents]} format matching existing usage?
            # Wait, CausalDiscovery.discover_causal_structure returned adjacency list {parent: [children]}?
            # PC algorithm returns undirected/directed graph.
            # Let's check previous code.
            # CausalDiscovery.pc_algorithm returns `graph = {var: []}` where list contains children?
            # "graph[variable_names[i]].append(variable_names[j])" -> adjacency_matrix[i,j]=1
            # Usually adj[i,j]=1 means i->j.
            # Let's verify standard.
            # My logic above constructed {child: [parents]}.
            # I should invert it to {parent: [children]} to match standard "Causal Graph" format if that's what's expected.
            # Or clarify return dict.
            # LabelBasedLayer2 uses it to find parents of 'target'.
            # It calls `sharpe_parents = causal_graph.get('TARGET_Sharpe', [])`.
            # If the dict is {node: [parents]}, then `get('TARGET')` returns parents.
            # If the dict is {node: [children]}, then `get('TARGET')` returns children.
            # Let's check `_run_causal_discovery` in `label_based_layer_2.py`:
            # "Extract parents of TARGET_Sharpe ... sharpe_parents = causal_graph.get('TARGET_Sharpe', [])"
            # AND "discovery_results = quick_bayesian_causal_discovery(...)"
            # "consensus_graph": {var: [parents]} usually for Bayesian networks.
            # BUT `CausalDiscovery.pc_algorithm` code showed: `graph[variable_names[i]].append(variable_names[j])`.
            # This looks like `i -> j`. So {parent: [children]}.
            # However, `LabelBasedLayer2` code:
            # "sharpe_parents = causal_graph.get('TARGET_Sharpe', [])"
            # If the graph was parent->children, getting 'TARGET' would give its children.
            # Context implies we want drivers (parents).
            # The Bayesian discovery usually returns parents list.
            # Let's standardize on returning **{Child: [Parents]}** which is more useful for "What drives X?".
            # Wait, if `LabelBasedLayer2` expects {Child: [Parents]}, and PC returns {Parent: [Children]}, there is a mismatch.
            # Let's check `_run_causal_discovery` again.
            # It prints: "Identified {len} drivers of Sharpe Ratio".
            # Drivers = Parents.
            # So `causal_graph` must be {Child: [Parents]}.
            # My `Huber` implementation produces {Child: [Parents]} naturally.
            # So I will return `causal_graph` as is (Child -> Parents).

            # Additional metadata
            'stability_stats': stability_stats,
            'causal_strength': causal_strength
        }

    def save_checkpoints(self, graph: Dict, stability: Dict):
        """Persist discovery artifacts."""
        try:
            out_dir = "outcomes/causal_discovery"
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save graph
            with open(f"{out_dir}/huber_graph_{ts}.json", 'w') as f:
                json.dump(graph, f, indent=2)

            # Save stability
            # Convert stability keys to string if needed
            safe_stability = {k: {str(p): c for p, c in v.items()} for k, v in stability.items()}
            with open(f"{out_dir}/huber_stability_{ts}.json", 'w') as f:
                json.dump(safe_stability, f, indent=2)

            if self.verbose:
                tprint_info(f"   💾 Checkpoints saved to {out_dir}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save checkpoints: {e}")

# Convenience wrapper
def quick_huber_discovery(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    discoverer = HuberCausalDiscovery(**kwargs)
    return discoverer.discover_causal_structure(df)
