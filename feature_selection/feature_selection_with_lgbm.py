"""
Feature Selection with LightGBM.

Implements a 3-stage feature selection pipeline:
1. EWMA Spearman IC + Spearman IC stability analysis pre-filters
2. Hierarchical clustering (avoids collinearity, keep the best in class)
3. 2-step RFE with light then strong LGBM
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import fastcluster
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selector using EWMA IC stability analysis, hierarchical clustering,
    and LightGBM-based recursive feature elimination.
    """
    
    def __init__(self, target_n_features: int = 50, verbose: bool = True):
        """
        Initialize the feature selector.
        
        Args:
            target_n_features: Target number of features to select
            verbose: Whether to print progress messages
        """
        self.target_n_features = target_n_features
        self.verbose = verbose
        self.ic_stats: Optional[Dict[str, Any]] = None
        self.original_columns: Optional[pd.Index] = None
        self.selection_report: Optional[Dict[str, Any]] = None
        self.cluster_assignments: Optional[Dict[str, int]] = None
        self.cluster_histogram: Optional[Dict[int, int]] = None
        self.cluster_assignments_csv_path: Optional[str] = None
        # Default maximum allowed absolute correlation for cluster caps
        self.cluster_rho_max: float = 0.3
        
    def _log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            prefix = {
                "info": "📊",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
            }.get(level, "📊")
            print(f"{prefix} FeatureSelector: {message}")
            
    def select_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
    ) -> List[str]:
        """
        Select features using the 3-stage pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target variable (Series or DataFrame)
            feature_names: Optional list of feature names (for logging)
            target_name: Optional target name (for logging)
            
        Returns:
            List of selected feature names
        """
        # Validate inputs
        if X is None or X.empty:
            self._log("Input X is empty, returning empty list", "warning")
            return []
            
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        if y is None or len(y) == 0:
            self._log("Target y is empty, returning empty list", "warning")
            return []
            
        # Store original columns
        self.original_columns = X.columns.copy()
        
        self._log(f"Starting feature selection: {len(X.columns)} features, {len(X)} samples")
        if target_name:
            self._log(f"Target: {target_name}")
            
        # Handle NaN values in target
        valid_indices = y.notna()
        if not valid_indices.all():
            nan_count = (~valid_indices).sum()
            self._log(f"Dropping {nan_count} rows with NaN target values", "warning")
            X = X.loc[valid_indices].copy()
            y = y.loc[valid_indices].copy()
            
        if len(X) == 0:
            self._log("No valid samples after NaN removal", "error")
            return []
            
        # Convert to float32 for efficiency
        try:
            X = X.astype(np.float32)
            y = y.astype(np.float32)
        except Exception as e:
            self._log(f"Error converting to float32: {e}", "warning")
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                self._log("No numeric columns found", "error")
                return []
            X = X[numeric_cols].astype(np.float32)

        prefilter_input_count = len(X.columns)
    
        # Pre-filter: keep only last 6 months of data if longer
        bars_per_day = 24 * 4  # 15-min bars
        bars_per_month = 30 * bars_per_day
        bars_6_months = 6 * bars_per_month
    
        if X.shape[0] > bars_6_months:
            self._log(f"Trimming to last 6 months: {bars_6_months} samples")
            X = X.iloc[-bars_6_months:]
            y = y.iloc[-bars_6_months:]
    
        # Stage 1: EWMA-based Pre-Filters
        self._log("Stage 1: Running EWMA IC stability pre-filters...")
        X_filtered = self.run_pre_filters(X, y)
        
        if X_filtered is None or X_filtered.empty:
            self._log("No features passed pre-filters, returning all original features", "warning")
            return list(self.original_columns[:self.target_n_features])
            
        stage1_count = len(X_filtered.columns)
        dropped_stage1 = prefilter_input_count - stage1_count
        self._log(
            f"Stage 1 summary: kept {stage1_count} / {prefilter_input_count} features "
            f"(dropped {dropped_stage1})"
        )
        self._log(f"Stage 1 complete: {stage1_count} features passed pre-filters")
    
        # Stage 2: Hierarchical Clustering
        self._log("Stage 2: Running hierarchical clustering to remove redundancy...")
        X_clustered = self._hierarchical_clustering(X_filtered, y)
        
        if X_clustered is None or X_clustered.empty:
            self._log("Clustering failed, using filtered features", "warning")
            X_clustered = X_filtered
            
        stage2_count = len(X_clustered.columns)
        dropped_stage2 = stage1_count - stage2_count
        self._log(
            f"Stage 2 summary: kept {stage2_count} / {stage1_count} features "
            f"(dropped {dropped_stage2})"
        )
        self._log(f"Stage 2 complete: {stage2_count} features after clustering")
    
        # Stage 3: LGBM-Based RFE
        self._log("Stage 3: Running LGBM RFE...")
        selected_features = self._lgbm_rfe(X_clustered, y)
        
        if not selected_features:
            self._log("LGBM RFE returned no features, using clustered features", "warning")
            selected_features = list(X_clustered.columns[:self.target_n_features])
            
        final_count = len(selected_features)
        dropped_stage3 = stage2_count - final_count
        self._log(
            f"Stage 3 summary: kept {final_count} / {stage2_count} features "
            f"(dropped {dropped_stage3})"
        )
        self._log(f"Stage 3 complete: {final_count} features selected", "success")

        # Optional: build cluster histogram for the final selection
        cluster_histogram: Dict[int, int] = {}
        cluster_assignments_selected: Dict[str, int] = {}
        cluster_csv_path: Optional[str] = None
        try:
            full_assignments = getattr(self, "cluster_assignments", None)
            cluster_csv_path = getattr(self, "cluster_assignments_csv_path", None)
            if isinstance(full_assignments, dict):
                for f in selected_features:
                    cid = full_assignments.get(f)
                    if isinstance(cid, int):
                        cluster_assignments_selected[f] = cid
                        cluster_histogram[cid] = cluster_histogram.get(cid, 0) + 1
        except Exception as e:
            self._log(f"Error building cluster histogram: {e}", "warning")

        # Generate report
        self._generate_report(X_filtered)

        # Store selection report for later inspection
        self.selection_report = {
            'total_input_features': len(self.original_columns),
            'after_prefilter': len(X_filtered.columns) if X_filtered is not None else 0,
            'after_clustering': len(X_clustered.columns) if X_clustered is not None else 0,
            'final_selected': len(selected_features),
            'selected_features': selected_features,
            'target_name': target_name,
            # Stage-wise drop counts for reporting
            'prefilter_input': prefilter_input_count,
            'stage1_kept': stage1_count,
            'stage1_dropped': dropped_stage1,
            'stage2_kept': stage2_count,
            'stage2_dropped': dropped_stage2,
            'stage3_kept': final_count,
            'stage3_dropped': dropped_stage3,
            # Correlation-cluster diagnostics
            'cluster_assignments_selected': cluster_assignments_selected,
            'cluster_histogram': cluster_histogram,
            'cluster_assignments_csv_path': cluster_csv_path,
        }

        return selected_features

    def run_pre_filters(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run pre-filters: remove constant features, high-NaN features, and rank by IC stability.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Filtered DataFrame with stable, informative features
        """
        # --- 1. Remove constant/near-constant features ---
        try:
            unique_counts = X.nunique()
            dominant_ratios = X.apply(lambda col: col.value_counts(normalize=True).max() if col.notna().sum() > 0 else 1.0)
            to_drop = X.columns[
                (unique_counts <= 1) |
                (dominant_ratios > 0.99)
            ]
            if len(to_drop) > 0:
                self._log(f"Dropping {len(to_drop)} constant/near-constant features")
                X = X.drop(columns=to_drop)
        except Exception as e:
            self._log(f"Error in constant feature detection: {e}", "warning")
    
        # --- 2. Remove features with > 5% NaN ---
        try:
            nan_ratios = X.isnull().sum() / len(X)
            high_nan_cols = nan_ratios[nan_ratios > 0.05].index
            if len(high_nan_cols) > 0:
                self._log(f"Dropping {len(high_nan_cols)} features with >5% NaN")
                X = X.drop(columns=high_nan_cols)
        except Exception as e:
            self._log(f"Error in NaN filtering: {e}", "warning")

        if X.shape[1] == 0:
            return X
    
        # --- 3. EWMA-based IC + stability analysis ---
        T_ic_window = 672  # 15-min bars per IC window (1 week)
        K = 12             # number of windows
        hl_days = 60       # half-life in days
        bars_per_day = 24 * 4  # 15-min bars
        hl_samples = hl_days * bars_per_day
        alpha = 1 - np.exp(-np.log(2) / hl_samples)
        eps = 1e-9
    
        n_samples = X.shape[0]
        ic_series = []
    
        for i in range(K):
            start = n_samples - (K - i) * T_ic_window
            end = start + T_ic_window
            if start < 0:
                continue
            X_window = X.iloc[start:end]
            y_window = y.iloc[start:end]
            
            # Skip windows with insufficient data
            if len(X_window) < 100:
                continue
                
            ic = self._calculate_spearman_correlation(X_window, y_window)
            ic_series.append(ic)
    
        ic_series = np.array(ic_series)
        if ic_series.shape[0] == 0:
            self._log("Not enough data for IC analysis, returning all features", "warning")
            return X
    
        # EWMA IC calculations
        ic_ewma = pd.DataFrame(ic_series).ewm(alpha=alpha, adjust=False).mean().iloc[-1].values
        ic_ewm_var = pd.DataFrame(ic_series).ewm(alpha=alpha, adjust=False).var().iloc[-1].values
        ic_ewm_std = np.sqrt(np.maximum(ic_ewm_var, 0))  # Ensure non-negative
    
        # Stability metrics
        ewma_sharpe = ic_ewma / (ic_ewm_std + eps)
        cv = ic_ewm_std / (np.abs(ic_ewma) + eps)
        positivity = (ic_series > 0).mean(axis=0)
    
        # Adaptive CUSUM
        cusum = np.cumsum(ic_series - ic_ewma, axis=0)
        cusum_recent = np.max(np.abs(cusum[-min(5, len(cusum)):]), axis=0)
        cusum_weight = min(0.5, cusum_recent.std() / (cusum_recent.mean() + eps))
    
        # Normalize metrics
        ewma_sharpe_norm = pd.Series(ewma_sharpe, index=X.columns).rank(pct=True)
        cv_norm = pd.Series(cv, index=X.columns).rank(pct=True, ascending=False)
        positivity_norm = pd.Series(positivity, index=X.columns).rank(pct=True)
        cusum_norm = pd.Series(cusum_recent, index=X.columns).rank(pct=True, ascending=False)
    
        # Stability score
        stability_score = ewma_sharpe_norm - 0.8 * cv_norm + 0.5 * positivity_norm - cusum_weight * cusum_norm
    
        # Normalize EWMA IC
        ic_norm = pd.Series(ic_ewma, index=X.columns).rank(pct=True)

        # Calculate IC volatility 
        ic_volatility = np.std(ic_ewma) / (np.mean(np.abs(ic_ewma)) + eps)
        max_ic_weight = 0.7
        min_ic_weight = 0.3
        ic_volatility = np.clip(ic_volatility, 0, 1)
        
        # Map to IC weight: higher volatility → lower IC weight
        w_ic = max_ic_weight - (max_ic_weight - min_ic_weight) * ic_volatility
        w_stability = 1 - w_ic

        # Combined score: stability + IC
        combined_score = w_stability * stability_score + w_ic * ic_norm
    
        self.ic_stats = {
            'stability_score': stability_score.values,
            'ic_ewma': ic_ewma,
            'ewma_sharpe': ewma_sharpe,
            'cv': cv,
            'positivity': positivity,
            'cusum_recent': cusum_recent,
            'combined_score': combined_score.values,
            'columns': X.columns.tolist(),
        }
    
        # --- 4. Percentile + hard cap ---
        percentile_cut = 0.2  # keep top 80%
        candidate_features = combined_score[combined_score.rank(pct=True) > percentile_cut]
        
        # Fallback if percentile cut yields too few candidates
        if len(candidate_features) < self.target_n_features:
            self._log(f"Percentile cut too aggressive ({len(candidate_features)} features), using all scored features", "warning")
            candidate_features = combined_score
    
        max_features = 5 * self.target_n_features
        n_keep = min(len(candidate_features), max_features)
    
        # Select top features by combined score
        stable_features = candidate_features.nlargest(n_keep).index
    
        return X[stable_features]

    def _hierarchical_clustering(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply hierarchical clustering to reduce redundancy by selecting the best
        feature from each cluster based on stability scores.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with cluster representatives only
        """
        if X is None or X.empty or len(X.columns) < 2:
            return X
            
        try:
            # Calculate rank correlation matrix
            ranked_X = rankdata(X.values, axis=0)
            corr = np.corrcoef(ranked_X.T)
            
            # Handle NaN in correlation matrix
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Distance matrix (1 - abs correlation)
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist, 0)  # Ensure diagonal is 0
            
            # Get condensed distance matrix (upper triangle)
            condensed_dist = dist[np.triu_indices(dist.shape[0], k=1)]
            
            # Apply hierarchical clustering
            Z = fastcluster.linkage(condensed_dist, method='average')
            
        except Exception as e:
            self._log(f"Error in clustering: {e}", "warning")
            return X

        # Get stability scores for feature selection within clusters
        if self.ic_stats is not None and 'stability_score' in self.ic_stats:
            try:
                # Align stability scores with current columns
                all_cols = self.ic_stats.get('columns', [])
                all_scores = self.ic_stats['stability_score']
                stability_series = pd.Series(all_scores, index=all_cols)
                
                # Filter to current columns and handle missing
                stability_scores = stability_series.reindex(X.columns)
                
                # Fill NaN with median
                if stability_scores.isna().any():
                    median_score = stability_scores.median()
                    if pd.isna(median_score):
                        median_score = 0.0
                    stability_scores = stability_scores.fillna(median_score)
                    
            except Exception as e:
                self._log(f"Error aligning stability scores: {e}", "warning")
                stability_scores = pd.Series(1.0, index=X.columns)
        else:
            # Compute stability scores from scratch
            ic_list = []
            for i in range(5):
                idx = np.random.choice(X.shape[0], size=int(0.15 * X.shape[0]), replace=False)
                X_sub = X.iloc[idx]
                y_sub = y.iloc[idx]
                ic_matrix = self._calculate_spearman_correlation(X_sub, y_sub)
                ic_list.append(ic_matrix)
            ic_array = np.vstack(ic_list)
            ic_mean = ic_array.mean(axis=0)
            ic_std = ic_array.std(axis=0)
            stability_scores = pd.Series(ic_mean - 0.5 * ic_std, index=X.columns)

        # Determine optimal number of clusters
        # Use a tighter ratio to target_n_features for more aggressive deduplication
        target_ratio_min = 0.7
        target_ratio_max = 1.3
        t = 0.4
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            clusters = fcluster(Z, t, criterion='distance')
            n_clusters = len(np.unique(clusters))
            ratio = n_clusters / self.target_n_features
            
            if ratio > target_ratio_max:
                t += 0.03
            elif ratio < target_ratio_min:
                t -= 0.03
            else:
                break
            
            # Prevent t from going too extreme
            t = np.clip(t, 0.05, 0.95)
            iteration += 1

        self._log(f"Clustering created {n_clusters} clusters (target ratio: {ratio:.2f})")

        # Select best feature from each cluster
        cluster_mapping = pd.DataFrame({'feature': X.columns, 'cluster': clusters})
        representative_features = []
        
        for cluster_id in cluster_mapping['cluster'].unique():
            features_in_cluster = cluster_mapping[cluster_mapping['cluster'] == cluster_id]['feature']
            cluster_scores = stability_scores.loc[features_in_cluster]
            
            # Handle all-NaN clusters
            if cluster_scores.isna().all():
                best_feature = features_in_cluster.iloc[0]
            else:
                best_feature = cluster_scores.idxmax()
                
            representative_features.append(best_feature)

        self._log(f"Selected {len(representative_features)} cluster representatives")
        return X[representative_features]

    def _cluster_cap_by_correlation(
        self,
        X: pd.DataFrame,
        features: List[str],
        max_per_cluster: int = 2,
        rho_max: Optional[float] = None,
    ) -> List[str]:
        """Limit how many features are kept per correlation cluster.

        This operates purely on the correlation structure between features,
        independent of naming. We cluster the selected features using a
        distance of ``1 - |Spearman(·,·)|`` and then walk the importance
        ranking, allowing at most ``max_per_cluster`` features from each
        cluster.
        """
        if not features:
            return features

        try:
            X_sub = X[features].copy()
            if X_sub.shape[1] <= max_per_cluster:
                return features

            # Rank-transform to compute Spearman correlations
            ranked_X = rankdata(X_sub.values, axis=0)
            corr = np.corrcoef(ranked_X.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)

            # Distance matrix and condensed form
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist, 0)
            condensed = dist[np.triu_indices(dist.shape[0], k=1)]

            # Hierarchical clustering on the selected features only
            Z = fastcluster.linkage(condensed, method="average")

            # Orthogonality-first clustering: use a fixed distance threshold.
            # dist = 1 - |rho|, so a maximum allowed correlation rho_max corresponds
            # to t = 1 - rho_max. If no rho_max is provided, fall back to the
            # instance-level default (self.cluster_rho_max) or 0.3.
            if rho_max is None:
                rho_max = getattr(self, "cluster_rho_max", 0.3)
            try:
                rho_max = float(rho_max)
            except Exception:
                rho_max = 0.3
            # Clamp to a reasonable range
            if rho_max <= 0.0 or rho_max >= 1.0:
                rho_max = 0.3
            t = 1.0 - rho_max

            clusters = fcluster(Z, t, criterion="distance")
            n_clusters = len(np.unique(clusters))
            self._log(
                f"Cluster-based cap: formed {n_clusters} correlation clusters "
                f"with rho_max={rho_max:.2f}"
            )

            # Map feature -> cluster id
            cluster_map = {
                feat: int(clusters[i]) for i, feat in enumerate(X_sub.columns)
            }

            # Persist cluster assignments for diagnostics
            try:
                self.cluster_assignments = dict(cluster_map)

                # Build histogram over all features considered in this cap
                cluster_histogram: Dict[int, int] = {}
                for cid in cluster_map.values():
                    cluster_histogram[cid] = cluster_histogram.get(cid, 0) + 1
                self.cluster_histogram = cluster_histogram

                # Optionally persist assignments to CSV for offline analysis
                try:
                    os.makedirs("outcomes", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = os.path.join(
                        "outcomes", f"feature_cluster_assignments_{ts}.csv"
                    )
                    pd.DataFrame(
                        {
                            "feature": list(cluster_map.keys()),
                            "cluster_id": list(cluster_map.values()),
                        }
                    ).to_csv(csv_path, index=False)
                    self.cluster_assignments_csv_path = csv_path
                    self._log(f"Cluster assignments saved to {csv_path}")
                except Exception as e_csv:
                    self._log(f"Error saving cluster assignments CSV: {e_csv}", "warning")
            except Exception as e_assign:
                self._log(f"Error recording cluster assignments: {e_assign}", "warning")

            # Walk features in ranked order, allowing at most max_per_cluster
            # from each correlation cluster.
            cluster_counts: Dict[int, int] = {}
            selected: List[str] = []

            for f in features:
                cid = cluster_map.get(f)
                if cid is None:
                    selected.append(f)
                    continue
                count = cluster_counts.get(cid, 0)
                if count < max_per_cluster:
                    selected.append(f)
                    cluster_counts[cid] = count + 1

            # Never exceed the global target
            target = int(self.target_n_features)
            if len(selected) > target:
                selected = selected[:target]

            return selected

        except Exception as e:
            self._log(f"Error in cluster-based cap: {e}", "warning")
            # Fallback: simple top-N truncation
            return features[: int(self.target_n_features)]

    def _apply_semantic_diversity_cap(
        self,
        features: List[str],
        max_per_family: int = 2,
    ) -> List[str]:
        """Limit features from the same semantic family to enforce conceptual diversity.
        
        Features are grouped into "families" based on their name patterns. For example:
        - simple_returns_3, simple_returns_7, simple_returns_10 → family "simple_returns"
        - vectorbt_trend_strength_5, vectorbt_trend_strength_10 → family "vectorbt_trend_strength"
        
        This ensures we don't have too many variations of the same concept even if
        they are statistically uncorrelated (different lookback windows).
        
        Args:
            features: List of feature names (already ranked by importance)
            max_per_family: Maximum features to keep per semantic family
            
        Returns:
            Filtered list of features with semantic diversity enforced
        """
        if not features or max_per_family <= 0:
            return features
            
        import re
        
        def _extract_family(name: str) -> str:
            """Extract the semantic family from a feature name.
            
            Strategy:
            1. Remove trailing numeric suffixes (lookback windows, thresholds)
            2. Remove common variant suffixes (_base, _vwap, _trend_adj, etc.)
            3. Keep the core concept name
            """
            # Normalize to lowercase for matching
            name_lower = name.lower()
            
            # Pattern 1: Features ending with _N or _N_suffix (e.g., simple_returns_10_price_returns)
            # Extract base name before the first numeric parameter
            match = re.match(r'^([a-z_]+?)_(\d+(?:\.\d+)?)', name_lower)
            if match:
                base = match.group(1)
                # Clean up trailing underscores
                base = base.rstrip('_')
                return base
            
            # Pattern 2: Features with numeric parameters in the middle
            # e.g., vectorbt_acceleration_trend_strength_5_10_price_returns
            # → vectorbt_acceleration_trend_strength
            parts = name_lower.split('_')
            non_numeric_parts = []
            for part in parts:
                # Stop at first numeric part
                if re.match(r'^\d+(?:\.\d+)?$', part):
                    break
                non_numeric_parts.append(part)
            
            if non_numeric_parts:
                return '_'.join(non_numeric_parts)
            
            # Fallback: use the full name as its own family
            return name_lower
        
        # Group features by family
        family_counts: Dict[str, int] = {}
        selected: List[str] = []
        skipped_by_family: Dict[str, List[str]] = {}
        
        for f in features:
            family = _extract_family(f)
            count = family_counts.get(family, 0)
            
            if count < max_per_family:
                selected.append(f)
                family_counts[family] = count + 1
            else:
                # Track skipped features for logging
                if family not in skipped_by_family:
                    skipped_by_family[family] = []
                skipped_by_family[family].append(f)
        
        # Log what was filtered
        if skipped_by_family:
            total_skipped = sum(len(v) for v in skipped_by_family.values())
            self._log(
                f"Semantic diversity filter: removed {total_skipped} features "
                f"from {len(skipped_by_family)} over-represented families"
            )
            # Log top families that had features removed
            for family, skipped in sorted(
                skipped_by_family.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:5]:
                self._log(f"  - {family}: kept {max_per_family}, skipped {len(skipped)}")
        
        return selected

    def _calculate_spearman_correlation(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate Spearman correlation between each feature and target."""
        try:
            X_ranked = rankdata(X.values, axis=0)
            y_ranked = rankdata(y.values, axis=0)
            ranked_data = np.hstack([X_ranked, y_ranked.reshape(-1, 1)])
            corr_matrix = np.corrcoef(ranked_data, rowvar=False)
            return corr_matrix[:-1, -1]
        except Exception as e:
            self._log(f"Error calculating correlations: {e}", "warning")
            return np.zeros(X.shape[1])

    def _generate_report(self, X: pd.DataFrame) -> None:
        """Generate and save a feature selection report."""
        if self.ic_stats is None:
            return

        try:
            cols = self.ic_stats.get('columns', X.columns if X is not None else [])
            report_df = pd.DataFrame({
                'stability_score': self.ic_stats.get('stability_score', []),
                'ewma_sharpe': self.ic_stats.get('ewma_sharpe', []),
                'cv': self.ic_stats.get('cv', []),
                'positivity': self.ic_stats.get('positivity', []),
                'cusum_recent': self.ic_stats.get('cusum_recent', []),
            }, index=cols[:len(self.ic_stats.get('stability_score', []))])

            os.makedirs('outcomes', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join('outcomes', f'feature_selection_report_{timestamp}.csv')
            report_df.to_csv(report_path)
            self._log(f"Report saved to {report_path}")
        except Exception as e:
            self._log(f"Error generating report: {e}", "warning")

    def _lgbm_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Perform LGBM-based recursive feature elimination.
        
        Uses two stages:
        1. Fast RFE with shadow features to remove clearly uninformative features
        2. Thorough RFE to select the final target number of features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        if X is None or X.empty:
            return []
            
        features = list(X.columns)
        
        if len(features) == 0:
            return []
            
        # Handle case where we have fewer features than target
        if len(features) <= self.target_n_features:
            self._log(f"Already at or below target ({len(features)} <= {self.target_n_features})")
            # Still apply correlation-cluster caps so that no single correlation
            # cluster dominates the final set.
            features = self._cluster_cap_by_correlation(X, features)
            return features
            
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X[features], y, test_size=0.2, random_state=42
            )
        except Exception as e:
            self._log(f"Error in train/val split: {e}", "warning")
            return features[:self.target_n_features]

        # Determine if this is a classification or regression problem
        unique_values = set(y.dropna().unique())
        is_classification = unique_values.issubset({0, 1, 0.0, 1.0})

        # Stage 1: Fast RFE with shadow features
        try:
            X_train_shadow = X_train[features].copy()
            X_val_shadow = X_val[features].copy()
            n_shadow = min(20, len(features))
            
            for i in range(n_shadow):
                col_name = f'shadow_{i}'
                col_idx = i % len(features)
                X_train_shadow[col_name] = np.random.permutation(X_train_shadow.iloc[:, col_idx].values)
                X_val_shadow[col_name] = np.random.permutation(X_val_shadow.iloc[:, col_idx].values)

            if is_classification:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    boosting_type='goss',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_samples=50,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                    top_rate=0.1,
                    other_rate=0.2
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    metric='rmse',
                    boosting_type='goss',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_samples=50,
                    min_child_weight=1e-3,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=2,
                    verbosity=-1,
                    top_rate=0.1,
                    other_rate=0.2
                )
                
            model.fit(
                X_train_shadow, y_train,
                eval_set=[(X_val_shadow, y_val)],
                callbacks=[lgb.early_stopping(5, verbose=False)]
            )

            importances = pd.Series(model.feature_importances_, index=X_train_shadow.columns)
            shadow_cols = [c for c in importances.index if c.startswith('shadow')]
            shadow_importance = importances[shadow_cols].mean() if shadow_cols else 0
            
            # Keep features that beat shadow importance
            surviving_features = importances[importances > shadow_importance].index
            features = [f for f in surviving_features if not f.startswith('shadow')]
            
            self._log(f"Stage 1 (shadow RFE): {len(features)} features survived")
            
            # Fallback if shadow removal was too aggressive
            if len(features) < self.target_n_features:
                self._log("Shadow RFE too aggressive, falling back to original features", "warning")
                features = list(X.columns)
                
        except Exception as e:
            self._log(f"Error in shadow RFE: {e}", "warning")
            features = list(X.columns)

        # Stage 2: Thorough RFE
        if len(features) > self.target_n_features:
            try:
                if is_classification:
                    model = lgb.LGBMClassifier(
                        objective='binary',
                        metric='auc',
                        boosting_type='gbdt',
                        n_estimators=750,
                        learning_rate=0.04,
                        max_depth=6,
                        min_child_samples=50,
                        subsample=0.7,
                        colsample_bytree=0.5,
                        bagging_freq=1,
                        reg_alpha=0.5,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=2,
                        verbosity=-1
                    )
                else:
                    model = lgb.LGBMRegressor(
                        objective='regression',
                        metric='rmse',
                        boosting_type='gbdt',
                        n_estimators=750,
                        learning_rate=0.04,
                        max_depth=6,
                        min_child_samples=50,
                        min_child_weight=1e-3,
                        subsample=0.7,
                        colsample_bytree=0.5,
                        bagging_freq=1,
                        reg_alpha=0.5,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=2,
                        verbosity=-1
                    )
                    
                model.fit(
                    X_train[features], y_train,
                    eval_set=[(X_val[features], y_val)],
                    callbacks=[lgb.early_stopping(10, verbose=False)]
                )

                importances = pd.Series(model.feature_importances_, index=features)
                
                # Progressive RFE
                p0 = len(features)
                pt = self.target_n_features
                rfe_alpha = 0.4
                min_drop = 1

                while len(features) > pt:
                    remaining = len(features) - pt
                    fraction = rfe_alpha * (remaining / max(p0 - pt, 1))
                    n_to_drop = max(min_drop, int(np.ceil(fraction * remaining)))
                    
                    # Get least important features
                    least_important = importances.loc[features].nsmallest(n_to_drop).index
                    features = [f for f in features if f not in least_important]
                    
                self._log(f"Stage 2 (thorough RFE): {len(features)} features selected")
                    
            except Exception as e:
                self._log(f"Error in thorough RFE: {e}", "warning")
                features = features[:self.target_n_features]

        # Apply correlation-cluster cap to encourage diversity across groups of
        # highly correlated features in the final selection while preserving the
        # learned ranking as much as possible.
        features = self._cluster_cap_by_correlation(X, features)
        
        return features
