"""
De Prado Feature Selection Engine for Meta-Labeling.

This module implements Marcos López de Prado's approach to feature selection:
1. ONC (Optimal Number of Clusters) - Redundancy reduction via correlation clustering (Regime-Aware & Denoised)
2. Advanced MDI (Mean Decrease Impurity) - Predictive power via ExtraTrees/LGBM (Deflated & Shadow-Adjusted)
3. Root Node Proximity - Structural hierarchy via tree depth analysis

Optimized with Numba and parallel processing.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.cluster import FeatureAgglomeration, KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import iqr, rankdata, spearmanr
from joblib import Parallel, delayed
from functools import lru_cache

from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.deprado_numba import get_node_depths_numba, calculate_entropy_numba, spearman_corr_numba
from src.utils.statistics_numba import denoise_correlation, get_precision_matrix, partial_corr_from_precision

# Optional LightGBM support
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

class DePradoFeatureEngine:
    """
    Complete Feature Selection Engine integrating:
    1. ONC Clustering (Redundancy Filter) - Denoised & Silhouette optimized, Regime-Aware
    2. Advanced MDI (Gain/Cover - Power Filter) - Deflated, Shadow-Adjusted, Cluster-Stacked
    3. Root Proximity (Hierarchy Filter) - Numba optimized
    4. Information Coefficient (Predictive Filter) - Directional & Stable
    5. Shannon Entropy (Information Content Filter) - Numba optimized
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_clusters: int = 12,
        min_cluster_size: int = 2,
        random_state: int = 42,
        gain_weight: float = 0.4,
        depth_weight: float = 0.1,
        ic_weight: float = 0.2,
        entropy_weight: float = 0.1,
        stability_weight: float = 0.3,
        min_samples_leaf: int = 30,
        max_features: str = 'log2',
        # Enhanced MDI params
        use_lgbm: bool = True,
        lgbm_params: Optional[Dict[str, Any]] = None,
        use_group_mdi: bool = True,
        # Hardening params
        max_cluster_size: int = 30,
        topk_freq_threshold: float = 0.4,
        is_regression: Optional[bool] = None,
        # Financial Logic
        use_denoising: bool = True,
        use_partial_corr: bool = True,
        use_regime_clustering: bool = True,
        use_turnover_penalty: bool = True
    ):
        self.n_estimators = n_estimators
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.gain_weight = gain_weight
        self.depth_weight = depth_weight
        self.ic_weight = ic_weight
        self.entropy_weight = entropy_weight
        self.stability_weight = stability_weight
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        self.use_lgbm = use_lgbm
        self.lgbm_params = lgbm_params or {}
        self.use_group_mdi = use_group_mdi
        
        if self.use_lgbm and not LGBM_AVAILABLE:
            tprint_warning('⚠️ LightGBM not available; disabling DePrado use_lgbm')
            self.use_lgbm = False

        self.max_cluster_size = max_cluster_size
        self.topk_freq_threshold = topk_freq_threshold
        self.is_regression = is_regression
        
        # Financial Logic
        self.use_denoising = use_denoising
        self.use_partial_corr = use_partial_corr
        self.use_regime_clustering = use_regime_clustering
        self.use_turnover_penalty = use_turnover_penalty

        # LightGBM Defaults
        if self.use_lgbm:
            defaults = {
                'max_depth': 4,
                'reg_alpha': 5,
                'reg_lambda': 10,
                'min_split_gain': 1e-3,
                'colsample_bytree': 0.7,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'importance_type': 'gain',
                'verbose': -1,
                'objective': 'binary'
            }
            for k, v in defaults.items():
                if k not in self.lgbm_params:
                    self.lgbm_params[k] = v

        # State
        self.feature_stats_ = None
        self.selected_features_ = None
        self.cluster_labels_ = None
        self.optimal_n_clusters_ = None
        
        self.lgbm_metrics_ = {}
        
        # Caches using method decorators or simple dicts with management
        # Note: Class-level cache removed as requested to avoid leaks
        self._cache_hits = 0

    def _rank_normalize(self, s: pd.Series, invert: bool = False) -> pd.Series:
        """Rank (Percentile) Normalization."""
        if s.empty: return s

        # Handle NaNs: Mask them, rank valid, then fill
        vals = s.values
        mask = np.isnan(vals)

        if mask.all():
            return pd.Series(0.5, index=s.index, dtype=np.float32)

        # Scipy rankdata is faster
        # Rank only valid values
        valid_vals = vals[~mask]
        ranks = rankdata(valid_vals, method='average')
        norm_valid = (ranks - 1) / (len(valid_vals) - 1 + 1e-9)

        # Reconstruct
        out = np.full_like(vals, 0.5, dtype=np.float32) # Default 0.5 for NaNs
        out[~mask] = norm_valid

        if invert:
            out = 1.0 - out

        return pd.Series(out, index=s.index, dtype=np.float32)

    def _get_onc_clusters(self, X: pd.DataFrame, global_corr: pd.DataFrame) -> pd.Series:
        """
        Finds optimal feature clusters using Multi-criteria ONC.
        """
        tprint_info(f"🔍 Finding optimal feature clusters (ONC) - Denoising: {self.use_denoising}...")
        
        # Base ONC Logic
        def run_base_onc(features_subset):
            # Slicing from global correlation
            sub_corr = global_corr.loc[features_subset, features_subset].fillna(0)
            
            # Distance matrix
            dist = 1 - np.abs(sub_corr.values)
            np.fill_diagonal(dist, 0)
            dist = np.maximum(dist, 0)

            best_k, best_score = 2, -1
            best_labels = None
            
            n_feats = len(features_subset)
            max_k = max(2, min(self.max_clusters, n_feats // 2))
            if n_feats > 200: max_k = min(max_k, 8)
            
            for k in range(2, max_k + 1):
                try:
                    clusterer = FeatureAgglomeration(n_clusters=k, linkage='average', metric='precomputed')
                    clusterer.fit(dist)
                    labels = clusterer.labels_
                    
                    if len(np.unique(labels)) > 1:
                        # Silhouette Score
                        score = silhouette_score(dist, labels, metric='precomputed')
                        if score > best_score:
                            best_score, best_k = score, k
                            best_labels = labels
                except Exception:
                    continue

            if best_labels is None:
                return np.zeros(n_feats)
            return best_labels

        # Initial Clustering
        initial_labels = run_base_onc(X.columns)
        final_series = pd.Series(initial_labels, index=X.columns)

        # Recursive Splitting
        final_map = {}
        next_cluster_id = 0
        queue = []
        
        for cid in sorted(np.unique(initial_labels)):
            queue.append((cid, final_series[final_series == cid].index, 0))

        while queue:
            cid, members, depth = queue.pop(0)

            if len(members) <= self.max_cluster_size or depth >= 3:
                # Leaf or max depth
                if len(members) > self.max_cluster_size:
                    # Guardrail A: Force split KMeans (Streaming PCA if large)
                    try:
                        sub_corr = global_corr.loc[members, members]
                        
                        # Use IncrementalPCA if cluster is huge to save memory
                        if len(members) > 100:
                            pca = IncrementalPCA(n_components=2, batch_size=100)
                        else:
                            pca = PCA(n_components=2)
                            
                        X_pca = pca.fit_transform(sub_corr)
                        n_chunks = max(2, int(np.ceil(len(members) / self.max_cluster_size)))
                        kmeans = KMeans(n_clusters=n_chunks, random_state=self.random_state)
                        sub_labels = kmeans.fit_predict(X_pca)
                        
                        for sub_cid in np.unique(sub_labels):
                            sub_members = members[sub_labels == sub_cid]
                            for m in sub_members: final_map[m] = next_cluster_id
                            next_cluster_id += 1
                    except:
                        for m in members: final_map[m] = next_cluster_id
                        next_cluster_id += 1
                else:
                    for m in members: final_map[m] = next_cluster_id
                    next_cluster_id += 1
            else:
                # Recurse
                try:
                    sub_labels = run_base_onc(members)
                    if len(np.unique(sub_labels)) > 1:
                        for sub_cid in np.unique(sub_labels):
                            sub_members = members[sub_labels == sub_cid]
                            queue.append((sub_cid, sub_members, depth + 1))
                    else:
                        queue.append((cid, members, 100)) # Force leaf
                except:
                    queue.append((cid, members, 100))

        # Guardrail B: Merge tiny clusters
        temp_series = pd.Series(final_map)
        cluster_sizes = temp_series.value_counts()
        tiny_clusters = cluster_sizes[cluster_sizes < self.min_cluster_size].index.tolist()
        
        if tiny_clusters:
            valid_clusters = [c for c in cluster_sizes.index if c not in tiny_clusters]
            if not valid_clusters: valid_clusters = [cluster_sizes.idxmax()]
            
            for tiny_cid in tiny_clusters:
                tiny_members = [f for f, c in final_map.items() if c == tiny_cid]
                best_target = -1
                max_corr = -1.0
                
                for target_cid in valid_clusters:
                    target_members = [f for f, c in final_map.items() if c == target_cid]
                    avg_corr = global_corr.loc[tiny_members, target_members].mean().mean()
                    if avg_corr > max_corr:
                        max_corr = avg_corr
                        best_target = target_cid
                
                if best_target != -1:
                    for m in tiny_members: final_map[m] = best_target

        # Fix: Ensure indices are preserved when factorizing
        final_series = pd.Series(final_map)
        # Factorize returns (codes, uniques), we want codes
        codes, _ = pd.factorize(final_series)
        final_series = pd.Series(codes, index=final_series.index)
        
        self.optimal_n_clusters_ = final_series.nunique()
        return final_series

    def _get_tree_hierarchy_numba(self, model, feature_names: List[str]) -> pd.Series:
        """Calculates Mean First Split Depth using Numba."""
        n_features = len(feature_names)
        depth_sums = np.zeros(n_features, dtype=np.float32)
        depth_counts = np.zeros(n_features, dtype=np.int64)
        max_depth_overall = 0
        
        for tree in model.estimators_:
            t = tree.tree_
            max_depth_overall = max(max_depth_overall, t.max_depth)
            
            # Call Numba function
            feats, depths = get_node_depths_numba(
                t.children_left,
                t.children_right,
                t.feature,
                t.node_count
            )
            
            tree_min_depths = {}
            for f, d in zip(feats, depths):
                if f < n_features: 
                    if f not in tree_min_depths or d < tree_min_depths[f]:
                        tree_min_depths[f] = d
            
            for f, d in tree_min_depths.items():
                depth_sums[f] += d
                depth_counts[f] += 1
                
        mean_depths = {}
        for i, name in enumerate(feature_names):
            if depth_counts[i] > 0:
                mean_depths[name] = depth_sums[i] / np.float32(depth_counts[i])
            else:
                mean_depths[name] = max_depth_overall
                
        return pd.Series(mean_depths)

    def _process_fold_lgbm(self, fold_idx, train_idx, val_idx, X_np, y_np, feature_names, lgbm_params, metric, is_regression):
        """Helper for parallel processing of folds (Numpy based for speed)."""
        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]
        
        # Shadow Feature: Use multiple shadows for robustness
        n_samples_train = X_train.shape[0]
        n_samples_val = X_val.shape[0]
        n_shadows = 5
        
        # Add shadow columns
        shadow_train = np.random.normal(0, 1, size=(n_samples_train, n_shadows)).astype(np.float32)
        shadow_val = np.random.normal(0, 1, size=(n_samples_val, n_shadows)).astype(np.float32)
        
        X_train_shadow = np.hstack([X_train, shadow_train])
        X_val_shadow = np.hstack([X_val, shadow_val])
        
        # Train
        if is_regression:
            model = lgb.LGBMRegressor(**lgbm_params)
        else:
            model = lgb.LGBMClassifier(**lgbm_params)
            # Binarize if needed
            uniq = np.unique(y_train)
            if len(uniq) > 2:
                y_train = (y_train > 0).astype(int)
                y_val = (y_val > 0).astype(int)
        
        model.fit(
            X_train_shadow, y_train,
            feature_name='auto',
            categorical_feature='auto',
            eval_metric=metric
        )
        
        # Importance
        imp_raw = model.feature_importances_
        # Identify shadow importances (last n_shadows features)
        shadow_imps = imp_raw[-n_shadows:]
        # Use max shadow importance as threshold (Conservative)
        shadow_threshold = np.max(shadow_imps)
        
        # Zero out features worse than shadow
        imp_adj = imp_raw[:-n_shadows].copy() # Exclude shadows
        imp_adj[imp_adj <= shadow_threshold] = 0
        
        # Normalize
        imp_log = np.log1p(imp_adj)
        imp_sum = imp_log.sum()
        if imp_sum > 0:
            imp_norm = imp_log / imp_sum
        else:
            imp_norm = np.zeros_like(imp_log)
            
        fold_imp = pd.Series(imp_norm, index=feature_names, dtype=np.float32)
        
        # OOF Preds
        if is_regression:
            preds = model.predict(X_val_shadow)
        elif hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val_shadow)[:, 1]
        else:
            preds = model.predict(X_val_shadow)

        preds = preds.astype(np.float32)
            
        # Feature ICs (Numba Optimized Spearman)
        try:
            # Rank transform for Spearman (Optimized)
            # scipy.stats.rankdata with axis=0 is much faster than apply_along_axis
            X_val_ranked = rankdata(X_val, axis=0).astype(np.float32)
            y_val_ranked = rankdata(y_val).astype(np.float32)
            
            ics = spearman_corr_numba(X_val_ranked, y_val_ranked)
            feat_ics = dict(zip(feature_names, ics))
        except:
            feat_ics = {c: 0.0 for c in feature_names}
            
        return fold_imp, feat_ics, (val_idx, preds)

    def _compute_lgbm_importance_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Parallel CV for MDI."""
        if not LGBM_AVAILABLE: raise ImportError("LightGBM missing")
        
        tprint_info("🚀 Starting MDI OOF & Stability Analysis (Parallel)...")
        cv = PurgedKFoldTime(n_splits=5)
        
        metric = 'rmse' if self.is_regression else 'auc'
        feature_names = X.columns.tolist()
        
        # Convert to numpy once
        X_np = X.values.astype(np.float32)
        y_np = y.values
        
        # Parallel Execution
        results = Parallel(n_jobs=-1)(
            delayed(self._process_fold_lgbm)(
                i, train_idx, val_idx, X_np, y_np, feature_names,
                self.lgbm_params, metric, self.is_regression
            )
            for i, (train_idx, val_idx) in enumerate(cv.split(X))
        )
        
        fold_stats = []
        fold_feature_ics = []
        oof_preds_accum = pd.Series(np.nan, index=X.index, dtype=np.float32)
        
        for fold_imp, feat_ics, (val_idx, preds) in results:
            fold_stats.append(fold_imp)
            fold_feature_ics.append(feat_ics)
            oof_preds_accum.iloc[val_idx] = preds
            
        # Aggregation
        imp_df = pd.DataFrame(fold_stats).fillna(0.0).astype(np.float32)
        ic_df = pd.DataFrame(fold_feature_ics).fillna(0.0).astype(np.float32)
        
        # Deflated MDI
        mean_imp = imp_df.mean()
        std_imp = imp_df.std()
        n_folds = len(imp_df)
        deflated_imp = mean_imp - (std_imp / np.sqrt(n_folds))
        deflated_imp = deflated_imp.clip(lower=0).astype(np.float32)
        
        # Stability
        median_gain = imp_df.median()
        iqr_gain = pd.Series(iqr(imp_df.values, axis=0), index=imp_df.columns)
        stability = median_gain / (iqr_gain + 1e-9)
        
        # Directional IC & Stability
        mean_ic = ic_df.mean()
        # Sign agreement: % of folds with same sign as mean
        sign_agreement = (np.sign(ic_df) == np.sign(mean_ic)).mean()
        # Penalize unstable signs
        # If agreement is 1.0 -> penalty 1.0 (multiplier). If 0.5 (random) -> 0.0? 
        # User formula: score = signed * (1 - |disagreement|)
        # Disagreement = 1 - agreement?
        # Let's say agreement is 0.8 (4/5 folds match). Disagreement is 0.2. Multiplier 0.8.
        # If agreement is 0.6 (3/5). Disagreement 0.4. Multiplier 0.6.
        # This effectively is just multiplier = agreement.
        score_ic_raw = mean_ic * sign_agreement
        
        # Top-K
        k = max(1, int(len(imp_df.columns) * 0.2))
        topk_freq = imp_df.apply(lambda x: x >= x.nlargest(k).min(), axis=1).mean()
        
        # OOF IC
        valid = oof_preds_accum.notna() & y.notna()
        if valid.sum() > 10:
            oof_ic, _ = spearmanr(y[valid], oof_preds_accum[valid])
        else:
            oof_ic = 0.0
            
        return {
            'mean_gain': deflated_imp,
            'stability': stability,
            'oof_ic': oof_ic,
            'directional_ic': score_ic_raw, # Signed and penalized
            'feature_oof_ic': mean_ic,      # Raw mean IC
            'topk_freq': topk_freq
        }

    def run_selection(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None, use_entropy_as_king: bool = False) -> List[str]:
        tprint_info(f"🚀 Starting De Prado Feature Selection Engine (Faster & Robust)...")
        
        # 0. Setup
        if self.is_regression is None:
            self.is_regression = pd.api.types.is_float_dtype(y) and y.nunique() > 10
            
        # Convert X to Numpy Early
        X_np = X.values.astype(np.float32)
        feature_names = X.columns.tolist()
        
        # 1. Global Correlation (Optimized)
        tprint_info("📊 Computing Global Correlation...")
        
        # Lazy Sampling if too large
        if len(X) > 10000 and len(feature_names) > 500:
            tprint_info("   sampling rows for correlation...")
            idx = np.random.choice(len(X), 5000, replace=False)
            X_corr_input = X.iloc[idx]
        else:
            X_corr_input = X

        global_corr = X_corr_input.corr().fillna(0).astype(np.float32)
        
        # Regime-Aware Correlation (Conservative)
        if self.use_regime_clustering:
            tprint_info("   🔄 Regime-Aware Clustering Enabled...")
            # Simple Volatility Regime: Split by Median Volatility of rows
            # Approx volatility of sample: std dev across features? Or time?
            # Better: If we assume time-series, use rolling vol of first PC or Mean feature
            proxy_vol = X_corr_input.std(axis=1) # Cross-sectional dispersion as proxy?
            # Or just split time in half? User said "volatility/risk regime".
            # Let's try rolling volatility of the mean feature
            mean_feat = X_corr_input.mean(axis=1)
            # Cannot do rolling on random sample.
            # If we used X full:
            if len(X) == len(X_corr_input):
                if groups is not None:
                    # Group-aware rolling
                    roll_vol = mean_feat.groupby(groups).rolling(20).std().reset_index(0, drop=True)
                    # Realign to X index if needed
                    if not roll_vol.index.equals(mean_feat.index):
                        roll_vol = roll_vol.reindex(mean_feat.index)
                else:
                    roll_vol = mean_feat.rolling(20).std()

                median_vol = roll_vol.median()
                mask_high = roll_vol > median_vol
                
                if mask_high.sum() > 100:
                    corr_high = X[mask_high].corr().fillna(0)
                    corr_low = X[~mask_high].corr().fillna(0)

                    # Conservative Merge: shrinkage towards zero
                    # We keep the correlation only if it's stable across regimes.
                    c_h = corr_high.values
                    c_l = corr_low.values

                    # Sign agreement mask
                    sign_agree = np.sign(c_h) == np.sign(c_l)

                    # Min magnitude (if disagree, 0 is min magnitude effectively)
                    min_mag = np.minimum(np.abs(c_h), np.abs(c_l))

                    # If signs agree, use sign * min_mag. If not, use 0.
                    final_vals = np.where(sign_agree, np.sign(c_h) * min_mag, 0.0)

                    global_corr = pd.DataFrame(final_vals, index=global_corr.index, columns=global_corr.columns)
        
        if self.use_denoising:
            tprint_info("   🧹 Denoising Correlation Matrix (Marchenko-Pastur)...")
            q = X.shape[0] / X.shape[1]
            global_corr = denoise_correlation(global_corr, q)
            
        # 2. Clustering
        self.cluster_labels_ = self._get_onc_clusters(X, global_corr)
        
        # 3. MDI
        lgbm_res = {}
        if self.use_lgbm:
            lgbm_res = self._compute_lgbm_importance_cv(X, y)
            gain = lgbm_res['mean_gain']
        else:
            model = ExtraTreesRegressor if self.is_regression else ExtraTreesClassifier
            est = model(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
            est.fit(X, y)
            gain = pd.Series(est.feature_importances_, index=X.columns, dtype=np.float32)
            
        # 4. Hierarchy (Numba Optimized)
        tprint_info("🌳 Analysis: Structure & Entropy...")
        et_structure = ExtraTreesClassifier(n_estimators=50, max_features='sqrt', max_depth=5, n_jobs=-1)
        if self.is_regression:
            et_structure = ExtraTreesRegressor(n_estimators=50, max_features='sqrt', max_depth=5, n_jobs=-1)
        et_structure.fit(X, y)
        depth = self._get_tree_hierarchy_numba(et_structure, feature_names)
        
        # 5. Entropy (Numba Optimized)
        # Pre-compute bin edges for entropy
        n_bins = 10
        # Quantile binning
        # We need bin edges: (n_bins+1, n_features)
        # np.nanquantile might be slow on large X.
        # Use simple linspace min/max for speed if large, or percentile on sample
        if len(X) > 10000:
            subs = X_np[np.random.choice(len(X), 5000, replace=False)]
        else:
            subs = X_np
            
        # Calculate edges
        q_vals = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(subs, q_vals, axis=0).astype(np.float32) # (n_bins+1, n_features)
        
        ent_vals = calculate_entropy_numba(X_np, bin_edges)
        ent_series = pd.Series(ent_vals, index=X.columns)
        
        # 6. IC & Turnover
        score_ic_raw = lgbm_res.get('directional_ic', X.corrwith(y).fillna(0))
        # Turnover Penalty
        if self.use_turnover_penalty:
            # Calculate turnover (Mean Absolute Difference)
            # Normalize features first to make turnover comparable (Z-score)
            # Note: We use global mean/std. For huge data this creates a copy, but necessary for fair comparison.
            X_std = (X - X.mean()) / (X.std() + 1e-9)

            if groups is not None:
                # Group-aware differencing
                diffs = X_std.groupby(groups).diff().fillna(0).values
            else:
                # Global differencing
                diffs = X_std.diff().fillna(0).values

            turnover = np.mean(np.abs(diffs), axis=0)
            turnover_series = pd.Series(turnover, index=X.columns)
            score_turnover = self._rank_normalize(turnover_series, invert=True) # Low turnover -> High score
        else:
            score_turnover = pd.Series(0, index=X.columns)

        # 7. Cluster Stacked Importance
        cluster_gain_sum = gain.groupby(self.cluster_labels_).transform('sum')
        
        # 8. Scoring
        score_gain = self._rank_normalize(cluster_gain_sum)
        score_depth = self._rank_normalize(depth, invert=True)
        # IC Score: Magnitude of signed IC (which is already penalized for instability)
        score_ic = self._rank_normalize(score_ic_raw.abs())
        score_ent = self._rank_normalize(ent_series)
        score_stab = self._rank_normalize(lgbm_res.get('stability', pd.Series(0, index=X.columns)))
        
        composite = (
            self.gain_weight * score_gain +
            self.stability_weight * score_stab +
            self.ic_weight * score_ic +
            self.depth_weight * score_depth +
            self.entropy_weight * score_ent
        )
        
        if self.use_turnover_penalty:
             composite += 0.1 * score_turnover # Add turnover component

        self.feature_stats_ = pd.DataFrame({
            "Cluster": self.cluster_labels_,
            "Gain": gain,
            "StackedGain": cluster_gain_sum,
            "CompositeScore": composite
        })
        
        # 9. Selection
        tprint_info("👑 Selection...")
        selected = []
        
        prec_matrix = None
        if self.use_partial_corr:
            prec_matrix = get_precision_matrix(global_corr.values)
        
        for cid in sorted(self.cluster_labels_.unique()):
            cluster_feats = self.feature_stats_[self.feature_stats_["Cluster"] == cid]
            if cluster_feats.empty: continue
            
            ranked = cluster_feats.sort_values("CompositeScore", ascending=False)
            primary = ranked.index[0]
            selected.append(primary)
            
            if len(ranked) > 1:
                secondary = ranked.index[1]
                score_ratio = ranked.iloc[1]["CompositeScore"] / ranked.iloc[0]["CompositeScore"]
                if score_ratio > 0.95:
                    if self.use_partial_corr and prec_matrix is not None:
                        idx_p = X.columns.get_loc(primary)
                        idx_s = X.columns.get_loc(secondary)
                        p_corr = partial_corr_from_precision(prec_matrix, idx_p, idx_s)
                        is_orthogonal = abs(p_corr) < 0.3
                    else:
                        is_orthogonal = abs(global_corr.loc[primary, secondary]) < 0.7
                        
                    if is_orthogonal:
                        selected.append(secondary)
                        
        self.selected_features_ = selected
        tprint_success(f"✅ Selected {len(selected)} features from {self.optimal_n_clusters_} clusters.")
        return selected

    def get_feature_stats(self) -> pd.DataFrame:
        if self.feature_stats_ is None: raise ValueError("Run selection first")
        return self.feature_stats_.copy()
        
    def get_selected_features(self) -> List[str]:
        if self.selected_features_ is None: raise ValueError("Run selection first")
        return self.selected_features_.copy()

def de_prado_feature_selection(X, y, groups=None, **kwargs):
    engine = DePradoFeatureEngine(**kwargs)
    selected = engine.run_selection(X, y, groups=groups)
    return X[selected], engine
