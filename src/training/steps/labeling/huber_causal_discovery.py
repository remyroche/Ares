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
6.  **Strict Stability:** Enforces sign consistency and out-of-sample validation per split.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import warnings
from collections import defaultdict
import os
import json
import math
import time
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
                 max_lag: int = 2,
                 n_splits: int = 7,
                 epsilon: float = 1.35,
                 alpha: float = 0.0001,
                 stability_threshold: float = 0.7,
                 sign_stability_threshold: float = 0.8,
                 detection_threshold: float = 0.05, # Normalized scale
                 validation_threshold: float = 0.05, # Min OOS correlation
                 target_variance_threshold: float = 1e-8,
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
            sign_stability_threshold: Fraction of sign agreement required.
            detection_threshold: Minimum coefficient magnitude (std dev units) to count as an edge.
            validation_threshold: Minimum Out-of-Sample correlation to accept a split's model.
            n_jobs: Parallel jobs.
            verbose: Logging verbosity.
            target_variance_threshold: Minimum variance needed for targets/splits.
        """
        self.max_lag = max_lag
        self.n_splits = n_splits
        self.epsilon = epsilon
        self.alpha = alpha
        self.stability_threshold = stability_threshold
        self.sign_stability_threshold = sign_stability_threshold
        self.detection_threshold = detection_threshold
        self.validation_threshold = validation_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.target_variance_threshold = target_variance_threshold
        self.excluded_target_prefixes = (
            "open", "close", "high", "low", "volume", "raw__volume",
            "denoised_close", "raw__px", "volatility_"
        )

        # Artifact storage
        self.discovery_artifacts = {}

    def _fit_and_validate_split(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                predictor_names: List[str]) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Fit Huber on Train, Validate on Test.
        Returns coefficients and OOS correlation.
        """
        diagnostics = {
            "status": "ok",
            "oos_score": 0.0,
        }

        y_train_std = float(np.std(y_train)) if len(y_train) else 0.0
        diagnostics["train_std"] = y_train_std

        if len(y_train) < 50:
            diagnostics["status"] = "train_too_small"
            return {}, 0.0, diagnostics  # Too few samples

        # Early check for constant target
        if y_train_std < 1e-9:
            diagnostics["status"] = "constant_train_target"
            # Optional: tprint_warning if you want to keep seeing it, but usually this is just noise
            return {}, 0.0, diagnostics

        try:
            # Scale per split to prevent leakage
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Fit Huber
            model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, fit_intercept=True, max_iter=200)
            model.fit(X_train_scaled, y_train_scaled)

            # Validate on Test
            if len(y_test) > 10:
                diagnostics["test_std_raw"] = float(np.std(y_test))
                X_test_scaled = scaler_X.transform(X_test)
                y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

                y_pred = model.predict(X_test_scaled)

            # Check OOS correlation
                # Handle edge cases (constant pred)
                pred_std = float(np.std(y_pred))
                target_std = float(np.std(y_test_scaled))
                diagnostics["pred_std"] = pred_std
                diagnostics["target_std"] = target_std
                if pred_std < 1e-9 or target_std < 1e-9:
                    oos_score = 0.0
                    diagnostics["status"] = "constant_pred_or_target"
                    # Only log if verbose AND truly degenerate (reduces noise)
                    # if self.verbose:
                    #     tprint_warning(f"         ⚠️ Split validation skipped: pred_std={pred_std:.2e}, target_std={target_std:.2e}")
                else:
                    oos_score = np.corrcoef(y_test_scaled, y_pred)[0, 1]
                    diagnostics["oos_score"] = float(oos_score)
                    if abs(oos_score) < self.validation_threshold:
                        diagnostics["status"] = "low_oos_corr"
                    
                if self.verbose and abs(oos_score) < self.validation_threshold:
                     tprint_info(f"         📉 Split validation failed: OOS Corr {oos_score:.4f} < {self.validation_threshold}")

            else:
                oos_score = 0.0 # Cannot validate
                diagnostics["status"] = "test_too_small"
                diagnostics["target_std"] = 0.0
                if self.verbose:
                     tprint_warning(f"         ⚠️ Split validation skipped: Insufficient test samples ({len(y_test)})")

            # Extract significant coefs
            coeffs = {}
            for name, coef in zip(predictor_names, model.coef_):
                if abs(coef) > self.detection_threshold:
                    coeffs[name] = coef

            return coeffs, oos_score, diagnostics

        except Exception:
            diagnostics["status"] = "exception"
            return {}, 0.0, diagnostics

    def _process_target_variable(self, target_col: str, X_lagged: np.ndarray,
                                 y_target: np.ndarray, predictor_names: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Process a single target variable across all splits with validation."""

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        split_coeffs = []
        valid_splits = 0
        split_failures = defaultdict(int)

        y_target = np.asarray(y_target, dtype=float)
        y_nan_rate = float(np.isnan(y_target).mean()) if len(y_target) else 0.0
        y_std = float(np.nanstd(y_target)) if len(y_target) else 0.0
        y_var = float(np.nanvar(y_target)) if len(y_target) else 0.0
        finite_mask = np.isfinite(y_target) & np.isfinite(X_lagged).all(axis=1)
        cleaned_samples = int(finite_mask.sum())
        if cleaned_samples < max(50, self.n_splits + 1):
            return target_col, {
                'parents': [],
                'strengths': {},
                'valid_splits': 0,
                'diagnostics': {
                    'split_failures': {'insufficient_clean_samples': self.n_splits},
                    'target_nan_rate': y_nan_rate,
                    'target_std': y_std,
                    'target_var': y_var,
                    'cleaned_samples': cleaned_samples,
                },
            }

        X_lagged = X_lagged[finite_mask]
        y_target = y_target[finite_mask]

        flat_split_stats = []

        # Execute splits
        for split_idx, (train_index, test_index) in enumerate(tscv.split(X_lagged)):
            X_train, X_test = X_lagged[train_index], X_lagged[test_index]
            y_train, y_test = y_target[train_index], y_target[test_index]

            test_std = float(np.std(y_test)) if len(y_test) else 0.0
            if test_std < self.target_variance_threshold:
                split_failures["test_target_flat"] += 1
                flat_split_stats.append({
                    "split": split_idx,
                    "status": "test_target_flat",
                    "pred_std": None,
                    "target_std": test_std,
                    "train_std": float(np.std(y_train)) if len(y_train) else 0.0,
                })
                if self.verbose:
                    tprint_info(
                        f"         ⚠️ Split {split_idx} skipped: test target std {test_std:.2e} < {self.target_variance_threshold:.1e}"
                    )
                continue

            coeffs, oos_score, diag = self._fit_and_validate_split(
                X_train, y_train, X_test, y_test, predictor_names
            )

            # Only count splits where the model learned something generalizable
            if abs(oos_score) > self.validation_threshold:
                split_coeffs.append(coeffs)
                valid_splits += 1
            else:
                status = diag.get("status", "unknown")
                split_failures[status] += 1
                if status in {"constant_pred_or_target", "constant_train_target"}:
                    flat_split_stats.append({
                        "split": split_idx,
                        "status": status,
                        "pred_std": diag.get("pred_std"),
                        "target_std": diag.get("target_std"),
                        "train_std": diag.get("train_std"),
                    })

        # Aggregation / Stability Selection
        # Track counts, magnitudes, and signs
        edge_stats = defaultdict(lambda: {'count': 0, 'magnitudes': [], 'signs': []})

        for res in split_coeffs:
            for pred, coef in res.items():
                edge_stats[pred]['count'] += 1
                edge_stats[pred]['magnitudes'].append(abs(coef))
                edge_stats[pred]['signs'].append(np.sign(coef))

        # Filter by stability gates
        stable_parents = []
        parent_strengths = {}

        # Hard stability gate: ceil(threshold * n_splits)
        # Note: We use total n_splits for threshold, or valid_splits?
        # User said: "stability gate = >= 4/5 splits". Usually implies total.
        # But if validation fails, the edge wasn't stable/predictive.
        # So we stick to n_splits base.
        required_count = math.ceil(self.n_splits * self.stability_threshold)

        for pred, stats in edge_stats.items():
            count = stats['count']
            signs = np.array(stats['signs'])

            # 1. Existence Stability
            if count < required_count:
                continue

            # 2. Sign Consistency
            pos_count = np.sum(signs > 0)
            neg_count = np.sum(signs < 0)
            major_sign_count = max(pos_count, neg_count)
            sign_agreement = major_sign_count / count

            if sign_agreement < self.sign_stability_threshold:
                continue

            # Passed all gates
            stable_parents.append(pred)
            parent_strengths[pred] = np.mean(stats['magnitudes'])

        return target_col, {
            'parents': stable_parents,
            'strengths': parent_strengths,
            'valid_splits': valid_splits,
            'diagnostics': {
                'split_failures': dict(split_failures),
                'flat_split_count': len(flat_split_stats),
                'flat_split_examples': flat_split_stats[:5],
                'target_nan_rate': y_nan_rate,
                'target_std': y_std,
                'target_var': y_var,
                'cleaned_samples': cleaned_samples,
                'is_purely_flat': len(flat_split_stats) == self.n_splits
            },
        }

    def _inject_market_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'Market_Mode' proxy if no obvious market index exists."""
        # Simple heuristic: First Principal Component of all returns
        # This captures the common factor driving the system
        try:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            pca = PCA(n_components=1)
            market_mode = pca.fit_transform(df_scaled).flatten()

            # Orient market mode to be positively correlated with mean return
            # (PCA sign is arbitrary)
            mean_ret = np.mean(df_scaled, axis=1)
            corr = np.corrcoef(market_mode, mean_ret)[0, 1]
            if corr < 0:
                market_mode = -market_mode

            df_aug = df.copy()
            df_aug['Market_Mode_PCA'] = market_mode
            return df_aug
        except Exception:
            return df

    def discover_causal_structure(self, df: pd.DataFrame, target_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the discovery pipeline.

        Args:
            df: Input dataframe (numeric).
            target_variable: (Optional) Focus variable.

        Returns:
            Dictionary with 'causal_graph' (adjacency list) and metadata.
        """
        if df.empty:
            tprint_error("HuberCausalDiscovery: Empty dataframe.")
            return {}

        if self.verbose:
            tprint_info(f"🚀 Huber Causal Discovery: {df.shape[1]} variables, {len(df)} samples")
            tprint_info(f"   ⚙️ Config: lag={self.max_lag}, splits={self.n_splits}, epsilon={self.epsilon}")
            tprint_info(f"   🛡️ Stability: >{self.stability_threshold:.0%} freq, >{self.sign_stability_threshold:.0%} sign agree")

        # 1. Data Preparation
        # Clean FIRST to ensure Market Mode and checks work on valid data
        df_proc = df.replace([np.inf, -np.inf], np.nan)
        target_cols = [c for c in df_proc.columns if c.startswith('TARGET_')]
        feature_cols = [c for c in df_proc.columns if c not in target_cols]
        if feature_cols:
            df_proc[feature_cols] = df_proc[feature_cols].ffill().bfill()
        # Drop remaining NaNs (if any purely NaN columns exist)
        # We don't drop rows yet because we want to see column quality first?
        # Actually, ffill/bfill preserves length.

        # Inject Market Mode to capture confounders
        df_proc = self._inject_market_mode(df_proc)
        
        # Check for bad columns AFTER cleaning attempts
        nan_rates = df_proc.isna().mean()
        drop_high_nan = nan_rates[nan_rates > 0.1].index.tolist()
        if drop_high_nan and self.verbose:
            tprint_warning(f"   ⚠️ Dropping high-NaN columns (after ffill): {drop_high_nan[:5]}")
        if drop_high_nan:
            df_proc = df_proc.drop(columns=drop_high_nan)
            
        low_var_cols = df_proc.std().fillna(0.0)
        drop_low_var = low_var_cols[low_var_cols < 1e-9].index.tolist()
        if drop_low_var and self.verbose:
            tprint_warning(f"   ⚠️ Dropping low-variance columns: {drop_low_var[:5]}")
        if drop_low_var:
            df_proc = df_proc.drop(columns=drop_low_var)
            
        # Final cleanup
        df_proc = df_proc.dropna()

        # Create Lagged Features
        # We do NOT scale globally here. We scale inside splits.
        lagged_data = {}
        # Target data must align (drop first max_lag rows)
        target_data = df_proc.iloc[self.max_lag:].copy()

        predictor_names = []

        # Build lagged matrix
        for lag in range(1, self.max_lag + 1):
            df_shifted = df_proc.shift(lag).iloc[self.max_lag:]
            for col in df_shifted.columns:
                name = f"{col}_L{lag}"
                lagged_data[name] = df_shifted[col].values
                predictor_names.append(name)

        X_matrix = pd.DataFrame(lagged_data).values # (N_samples, N_vars * max_lag)

        if self.verbose:
            tprint_info(f"   📊 Feature Matrix: {X_matrix.shape} (lags construction)")

        # 2. Parallel Node-wise Regression
        column_stds = df_proc.std()
        excluded_targets = []
        targets = []
        for col in df_proc.columns:
            col_lower = col.lower()
            if any(col_lower.startswith(prefix) for prefix in self.excluded_target_prefixes):
                excluded_targets.append(col)
                continue
            if column_stds.get(col, 0.0) < self.target_variance_threshold:
                excluded_targets.append(col)
                continue
            targets.append(col)

        if self.verbose and excluded_targets:
            tprint_info(f"   ⏭️ Skipping {len(excluded_targets)} flat/core columns as discovery targets")

        if self.verbose:
            tprint_info(
                f"   🧵 Running node-wise regressions: targets={len(targets)}, predictors={len(predictor_names)}, "
                f"samples={X_matrix.shape[0]}, jobs={self.n_jobs}"
            )
        discovery_start = time.time()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_target_variable)(
                target, X_matrix, target_data[target].values, predictor_names
            ) for target in targets
        )
        if self.verbose:
            elapsed = time.time() - discovery_start
            tprint_info(f"   ⏱️ Node-wise regressions complete in {elapsed:.1f}s")

        # 3. Construct Graph
        causal_graph = {}
        causal_strength = {}
        valid_split_stats = {}
        diagnostics_by_target = {}

        total_edges = 0

        for target, res in results:
            parents_clean = set()
            strengths_clean = {}

            for pred_name in res['parents']:
                if "_L" in pred_name:
                    parent_var = pred_name.rsplit("_L", 1)[0]
                    # Self-loops allowed in lags (Autoregression)
                    # But if we want purely cross-sectional structure, we might filter.
                    # Usually autoregression is good to keep as it explains variance.

                    parents_clean.add(parent_var)

                    s = res['strengths'][pred_name]
                    if parent_var in strengths_clean:
                        strengths_clean[parent_var] = max(strengths_clean[parent_var], s)
                    else:
                        strengths_clean[parent_var] = s

            if parents_clean:
                causal_graph[target] = list(parents_clean)
                causal_strength[target] = strengths_clean

            valid_split_stats[target] = res['valid_splits']
            diagnostics_by_target[target] = res.get('diagnostics', {})
            total_edges += len(parents_clean)

        # 4. Persistence
        self.save_checkpoints(causal_graph, valid_split_stats)

        if self.verbose:
            tprint_success(f"✅ Huber Discovery Complete: {total_edges} lagged dependency edges found.")
            avg_valid = np.mean(list(valid_split_stats.values())) if valid_split_stats else 0
            tprint_info(f"   ℹ️ Avg Valid Splits: {avg_valid:.1f}/{self.n_splits}")
            if diagnostics_by_target:
                failure_counts = defaultdict(int)
                low_var_targets = []
                high_nan_targets = []
                for target, diag in diagnostics_by_target.items():
                    split_failures = diag.get("split_failures", {})
                    for key, val in split_failures.items():
                        failure_counts[key] += int(val)
                    if diag.get("target_std", 0.0) < 1e-6:
                        low_var_targets.append(target)
                    if diag.get("target_nan_rate", 0.0) > 0.1:
                        high_nan_targets.append(target)
                if failure_counts:
                    tprint_info(f"   🧪 Split failure summary: {dict(failure_counts)}")
                
                flat_targets = []
                purely_flat_count = 0
                
                for target, diag in diagnostics_by_target.items():
                    if diag.get("is_purely_flat"):
                        purely_flat_count += 1
                        continue
                        
                    flat_count = int(diag.get("flat_split_count", 0))
                    if flat_count:
                        examples = diag.get("flat_split_examples", [])
                        pred_stds = [e.get("pred_std") for e in examples if e.get("pred_std") is not None]
                        tgt_stds = [e.get("target_std") for e in examples if e.get("target_std") is not None]
                        train_stds = [e.get("train_std") for e in examples if e.get("train_std") is not None]
                        flat_targets.append((
                            target,
                            flat_count,
                            float(np.mean(pred_stds)) if pred_stds else None,
                            float(np.mean(tgt_stds)) if tgt_stds else None,
                            float(np.mean(train_stds)) if train_stds else None,
                        ))
                
                if purely_flat_count > 0:
                    tprint_info(f"   ⏭️ Skipped {purely_flat_count} targets causing pure flat splits (no variance).")

                if flat_targets:
                    flat_targets.sort(key=lambda x: x[1], reverse=True)
                    top_flat = flat_targets[:10]
                    summary = []
                    for name, count, pred_std, tgt_std, train_std in top_flat:
                        if pred_std is None or tgt_std is None or train_std is None:
                            summary.append(f"{name} ({count}, no-variance)")
                        else:
                            summary.append(
                                f"{name} ({count}, pred_std~{pred_std:.2e}, tgt_std~{tgt_std:.2e}, train_std~{train_std:.2e})"
                            )
                    if summary:
                        tprint_warning(f"   ⚠️ Flat split targets (partial): {summary}")
                if low_var_targets:
                    tprint_warning(f"   ⚠️ Low-variance targets: {low_var_targets[:5]}")
                if high_nan_targets:
                    tprint_warning(f"   ⚠️ High-NaN targets: {high_nan_targets[:5]}")

        return {
            'causal_graph': causal_graph,  # {Child: [Parents]}
            'causal_strength': causal_strength,
            'valid_split_stats': valid_split_stats,
            'diagnostics': diagnostics_by_target,
        }

    def save_checkpoints(self, graph: Dict, stats: Dict):
        """Persist discovery artifacts."""
        try:
            out_dir = "outcomes/causal_discovery"
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save graph
            with open(f"{out_dir}/lagged_dependency_skeleton_{ts}.json", 'w') as f:
                json.dump(graph, f, indent=2)

            if self.verbose:
                tprint_info(f"   💾 Lagged skeleton saved to {out_dir}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save checkpoints: {e}")

# Convenience wrapper
def quick_huber_discovery(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    discoverer = HuberCausalDiscovery(**kwargs)
    return discoverer.discover_causal_structure(df)
