"""
De Prado Feature Selection Engine for Meta-Labeling.

This module implements Marcos López de Prado's approach to feature selection:
1. ONC (Optimal Number of Clusters) - Redundancy reduction via correlation clustering
2. Advanced MDI (Mean Decrease Impurity) - Predictive power via ExtraTrees 
3. Root Node Proximity - Structural hierarchy via tree depth analysis

The engine selects one "king" feature from each cluster based on composite score:
Composite = 0.5 * Normalized_Gain + 0.5 * Normalized_Depth_Proximity

Usage:
- Reduces feature redundancy while preserving predictive power
- Ensures structural diversity in selected features
- Complements CMI filtering for orthogonal feature selection
"""

import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.cluster import FeatureAgglomeration, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import norm, entropy, spearmanr, iqr, rankdata
import warnings
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
from src.utils.purged_kfold import PurgedKFoldTime

# Optional LightGBM support
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

EPS = 1e-12

class DePradoFeatureEngine:
    """
    Complete Feature Selection Engine integrating:
    1. ONC Clustering (Redundancy Filter)
    2. Advanced MDI (Gain/Cover - Power Filter)  
    3. Root Proximity (Hierarchy Filter)
    4. Information Coefficient (Predictive Filter)
    5. Shannon Entropy (Information Content Filter)
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_clusters: int = 12,
        min_cluster_size: int = 2,
        random_state: int = 42,
        gain_weight: float = 0.4,
        depth_weight: float = 0.1,  # Reduced default
        ic_weight: float = 0.2,
        entropy_weight: float = 0.1,
        stability_weight: float = 0.3,
        min_samples_leaf: int = 30,
        max_features: str = 'log2',
        # New parameters for Enhanced MDI
        use_lgbm: bool = True,
        lgbm_params: Optional[Dict[str, Any]] = None,
        use_group_mdi: bool = True,
        # Hardening params
        max_cluster_size: int = 30,
        topk_freq_threshold: float = 0.4,  # Increased robustness
        is_regression: Optional[bool] = None
    ):
        """
        Initialize De Prado Feature Engine.
        """
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
        
        # Enhanced settings
        self.use_lgbm = use_lgbm
        self.lgbm_params = lgbm_params or {}
        self.use_group_mdi = use_group_mdi

        if self.use_lgbm and not LGBM_AVAILABLE:
            tprint_warning('⚠️ LightGBM not available; disabling DePrado use_lgbm')
            self.use_lgbm = False

        # Hardening
        self.max_cluster_size = max_cluster_size
        self.topk_freq_threshold = topk_freq_threshold

        # Regression support
        self.is_regression = is_regression

        # Defaults for LightGBM if used
        if self.use_lgbm:
            defaults = {
                'max_depth': 4,
                'reg_alpha': 5,      # More balanced L1
                'reg_lambda': 10,    # L2 for stability
                'min_split_gain': 1e-3,
                'colsample_bytree': 0.7,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'importance_type': 'gain',
                'verbose': -1
            }
            for k, v in defaults.items():
                if k not in self.lgbm_params:
                    self.lgbm_params[k] = v

        # Results storage
        self.feature_stats_ = None
        self.selected_features_ = None
        self.cluster_labels_ = None
        self.optimal_n_clusters_ = None
        self.silhouette_scores_ = None
        
        # Extended diagnostics
        self.lgbm_metrics_ = {
            'mean_gain': None,
            'stability': None,
            'oof_ic': 0.0,
            'feature_oof_ic': None,
            'topk_freq': None,
            'median_log_gain': None,
            'iqr_log_gain': None
        }

    def _get_onc_clusters(self, X: pd.DataFrame) -> pd.Series:
        """
        Finds optimal feature clusters using Multi-criteria ONC.
        Includes recursive re-clustering to prevent mega-clusters and merges tiny clusters.
        """
        tprint_info("🔍 Finding optimal feature clusters (Multi-criteria ONC)...")
        onc_start = time.time()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Precompute global correlation for later merging
        global_corr = X.corr().abs().fillna(0)

        # Base ONC Logic
        def run_base_onc(X_subset):
            # Compute correlation matrix
            corr = X_subset.corr().fillna(0)
            corr = np.clip(corr, -1, 1)
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist.values, 0)
            dist = np.maximum(dist, 0)

            best_k, best_composite_score = 2, -1
            scores_history = {}
            # Allow K up to N/2
            max_k = max(2, min(self.max_clusters, len(X_subset.columns) // 2))

            feature_sample_matrix = X_subset.T

            for k in range(2, max_k + 1):
                try:
                    clusterer = FeatureAgglomeration(n_clusters=k, linkage='average')
                    clusterer.fit(X_subset)
                    labels = clusterer.labels_

                    if len(np.unique(labels)) > 1:
                        cv_ratio = self._calculate_cv_ratio(feature_sample_matrix, labels)
                        dbi = davies_bouldin_score(feature_sample_matrix, labels)
                        silhouette = silhouette_score(dist, labels, metric='precomputed')

                        cv_score = min(cv_ratio / 5.0, 1.0)
                        dbi_score = 1.0 / (1.0 + dbi)
                        silhouette_score_norm = (silhouette + 1.0) / 2.0

                        # Simplified composite
                        composite_score = 0.5 * cv_score + 0.3 * dbi_score + 0.2 * silhouette_score_norm

                        if composite_score > best_composite_score:
                            best_composite_score, best_k = composite_score, k
                except Exception:
                    continue

            # Final Fit
            if best_k == 1:
                return np.zeros(len(X_subset.columns))

            final_clusterer = FeatureAgglomeration(n_clusters=best_k, linkage='average')
            final_clusterer.fit(X_subset)
            return final_clusterer.labels_

        # Initial Clustering
        initial_labels = run_base_onc(X)
        final_series = pd.Series(initial_labels, index=X.columns)

        # Recursive Splitting (Max Depth 3)
        final_map = {}
        next_cluster_id = 0

        # Queue for processing clusters: (cluster_id, member_indices, depth)
        queue = []
        for cid in sorted(np.unique(initial_labels)):
            queue.append((cid, final_series[final_series == cid].index, 0))

        while queue:
            cid, members, depth = queue.pop(0)

            if len(members) <= self.max_cluster_size or depth >= 3:
                # Leaf cluster or max depth reached

                # Check Guardrail A: Still oversized at cap?
                if len(members) > self.max_cluster_size:
                    # Force split using KMeans on PCA
                    try:
                        pca = PCA(n_components=2)
                        X_sub = X[members]
                        X_pca = pca.fit_transform(X_sub.T) # Features as samples
                        # Split into enough chunks to satisfy max size
                        n_chunks = max(2, int(np.ceil(len(members) / self.max_cluster_size)))
                        kmeans = KMeans(n_clusters=n_chunks, random_state=self.random_state)
                        sub_labels = kmeans.fit_predict(X_pca)

                        for sub_cid in np.unique(sub_labels):
                            sub_members = members[sub_labels == sub_cid]
                            for m in sub_members: final_map[m] = next_cluster_id
                            next_cluster_id += 1
                    except:
                        # Fallback: keep as is
                        for m in members: final_map[m] = next_cluster_id
                        next_cluster_id += 1
                else:
                    # Accept cluster
                    for m in members: final_map[m] = next_cluster_id
                    next_cluster_id += 1
            else:
                # Recurse
                try:
                    sub_labels = run_base_onc(X[members])
                    if len(np.unique(sub_labels)) > 1:
                        for sub_cid in np.unique(sub_labels):
                            sub_members = members[sub_labels == sub_cid]
                            queue.append((sub_cid, sub_members, depth + 1))
                    else:
                        # Cannot split further by ONC -> Fallback to KMeans force split
                        queue.append((cid, members, 100)) # Force guardrail A path
                except:
                    queue.append((cid, members, 100))

        # Guardrail B: Merge tiny clusters (size < min_cluster_size)
        temp_series = pd.Series(final_map)
        cluster_sizes = temp_series.value_counts()
        tiny_clusters = cluster_sizes[cluster_sizes < self.min_cluster_size].index.tolist()

        # Sort tiny clusters by size ascending (merge smallest first)
        tiny_clusters.sort(key=lambda c: cluster_sizes[c])

        merged_map = final_map.copy()
        valid_clusters = [c for c in cluster_sizes.index if c not in tiny_clusters]

        for tiny_cid in tiny_clusters:
            tiny_members = [f for f, c in merged_map.items() if c == tiny_cid]
            if not tiny_members: continue

            # Find best target cluster
            best_target = -1
            max_avg_corr = -1.0

            # If no valid clusters exist yet (all tiny), merge into largest tiny cluster
            targets = valid_clusters if valid_clusters else [c for c in cluster_sizes.index if c != tiny_cid]

            for target_cid in targets:
                target_members = [f for f, c in merged_map.items() if c == target_cid]
                if not target_members: continue

                # Calculate avg correlation between tiny group and target group
                # Sub-matrix of correlations
                sub_corr = global_corr.loc[tiny_members, target_members]
                avg_corr = sub_corr.mean().mean()

                if avg_corr > max_avg_corr:
                    max_avg_corr = avg_corr
                    best_target = target_cid

            if best_target != -1:
                # Merge
                for m in tiny_members:
                    merged_map[m] = best_target
                # If target was also tiny, it might now be valid, but we rely on iterative passes or simple logic
                # For simplicity in this robust implementation, we merge into 'best available'.

        final_series = pd.Series(merged_map)

        # Re-index to be 0..N
        final_series = pd.factorize(final_series)[0]
        final_series = pd.Series(final_series, index=X.columns)

        self.optimal_n_clusters_ = final_series.nunique()
        tprint_info(f"⏱️  ONC completed: {self.optimal_n_clusters_} clusters found (recursive + merged)")

        return final_series

    def _get_onc_clusters_for_fold(self, X_train: pd.DataFrame) -> pd.Series:
        """
        Finds optimal feature clusters using Multi-criteria ONC inside CV loop.
        Includes simple recursion for mega-clusters.
        """
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        def run_fast_onc(X_sub):
            corr = X_sub.corr().fillna(0)
            corr = np.clip(corr, -1, 1)
            dist = 1 - np.abs(corr)
            np.fill_diagonal(dist.values, 0)

            best_k, best_score = 2, -1
            max_k = max(2, min(self.max_clusters, len(X_sub.columns) // 2))

            feature_sample_matrix = X_sub.T

            for k in range(2, max_k + 1):
                try:
                    clusterer = FeatureAgglomeration(n_clusters=k, linkage='average')
                    clusterer.fit(X_sub)
                    if len(np.unique(clusterer.labels_)) > 1:
                        # Simple DBI proxy
                        dbi = davies_bouldin_score(feature_sample_matrix, clusterer.labels_)
                        score = 1.0 / (1.0 + dbi)
                        if score > best_score:
                            best_score, best_k = score, k
                except: continue

            # Final Fit
            final_clusterer = FeatureAgglomeration(n_clusters=best_k, linkage='average')
            final_clusterer.fit(X_sub)
            return final_clusterer.labels_

        # Initial
        labels = run_fast_onc(X_train)
        series = pd.Series(labels, index=X_train.columns)

        # Simple recursion (1 level deep only for speed in fold)
        final_map = {}
        next_id = 0
        for cid in sorted(np.unique(labels)):
            members = series[series == cid].index
            if len(members) > self.max_cluster_size:
                sub_labels = run_fast_onc(X_train[members])
                for sub_cid in np.unique(sub_labels):
                    sub_members = members[sub_labels == sub_cid]
                    for m in sub_members: final_map[m] = next_id
                    next_id += 1
            else:
                for m in members: final_map[m] = next_id
                next_id += 1

        return pd.Series(final_map, index=X_train.columns)

    def _calculate_cv_ratio(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate CV Ratio."""
        try:
            # Subsample for speed
            if X.shape[1] > 1000:
                X = X.iloc[:, :1000]

            overall_centroid = X.mean(axis=0)
            bcss = 0.0
            wcss = 0.0
            
            # Vectorized calculation possible but keeping simple for robustness
            df_X = pd.DataFrame(X)
            df_X['label'] = labels

            # WCSS
            wcss = df_X.groupby('label').apply(lambda x: ((x - x.mean())**2).sum().sum()).sum()

            # BCSS
            cluster_means = df_X.groupby('label').mean()
            counts = df_X['label'].value_counts()

            # Align
            cluster_means = cluster_means.loc[counts.index]

            # Distance from global mean
            diff_sq = (cluster_means - overall_centroid)**2
            bcss = (diff_sq.sum(axis=1) * counts).sum()
            
            if wcss > 0:
                return bcss / wcss
            return 0.0
            
        except Exception:
            return 0.0
    
    def _get_tree_hierarchy(self, model: Union[ExtraTreesClassifier, ExtraTreesRegressor], feature_names: List[str]) -> pd.Series:
        """Calculates Mean First Split Depth."""

        depths = {name: [] for name in feature_names}
        max_depth_overall = 0
        
        for tree in model.estimators_:
            t = tree.tree_
            max_depth_overall = max(max_depth_overall, t.max_depth)
            first_occurrence = {}
            
            def walk_node(node: int, current_depth: int):
                if t.feature[node] != -2:
                    feature_idx = t.feature[node]
                    if feature_idx not in first_occurrence:
                        first_occurrence[feature_idx] = current_depth
                    walk_node(t.children_left[node], current_depth + 1)
                    walk_node(t.children_right[node], current_depth + 1)
            
            walk_node(0, 0)
            
            for idx, depth in first_occurrence.items():
                if idx < len(feature_names):
                    depths[feature_names[idx]].append(depth)
        
        mean_depths = {}
        for name, depth_list in depths.items():
            if depth_list:
                mean_depths[name] = np.median(depth_list)
            else:
                mean_depths[name] = max_depth_overall
        
        return pd.Series(mean_depths)
    
    def _compute_advanced_mdi(self, model: Union[ExtraTreesClassifier, ExtraTreesRegressor], feature_names: List[str]) -> Dict[str, float]:
        """Compute Advanced MDI metrics."""

        gain_importances = model.feature_importances_
        cover_counts = np.zeros(len(feature_names))
        total_samples = 0
        
        for tree in model.estimators_:
            t = tree.tree_
            n_samples = t.n_node_samples
            for i in range(t.node_count):
                if t.feature[i] != -2:
                    feature_idx = t.feature[i]
                    if feature_idx < len(feature_names):
                        cover_counts[feature_idx] += n_samples[i]
            total_samples += n_samples[0]
        
        cover_importances = cover_counts / (total_samples + EPS)
        
        return {
            'gain': dict(zip(feature_names, gain_importances)),
            'cover': dict(zip(feature_names, cover_importances))
        }

    def _fit_transform_pca_groups(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], pd.Series]:
        """
        Fit PCA groups on Training data, transform both Train and Val.
        Returns:
            X_train_pca, X_val_pca, loadings_map, cluster_map
        """
        # 1. Cluster on Train only
        cluster_map = self._get_onc_clusters_for_fold(X_train)

        X_train_groups = pd.DataFrame(index=X_train.index)
        X_val_groups = pd.DataFrame(index=X_val.index)
        loadings_map = {}

        for cluster_id in sorted(cluster_map.unique()):
            features = cluster_map[cluster_map == cluster_id].index
            group_key_base = f"Cluster_{cluster_id}"

            if len(features) > 1:
                # Retain up to 3 PCs or 70% variance
                pca = PCA(n_components=min(3, len(features)))
                scaler = StandardScaler()

                # Fit on Train
                X_train_cluster_scaled = scaler.fit_transform(X_train[features])
                # Note: No sign flip here yet, but loadings attribution handles magnitude.
                # To be fully deterministic, we could enforce sign based on max loading.
                X_train_pcs = pca.fit_transform(X_train_cluster_scaled)

                # Transform Val
                X_val_cluster_scaled = scaler.transform(X_val[features])
                X_val_pcs = pca.transform(X_val_cluster_scaled)

                # Check explained variance
                expl_var = np.cumsum(pca.explained_variance_ratio_)
                n_pcs = np.searchsorted(expl_var, 0.70) + 1
                n_pcs = min(n_pcs, X_train_pcs.shape[1])

                # Store PCs
                for i in range(n_pcs):
                    col_name = f"{group_key_base}_PC{i+1}"
                    X_train_groups[col_name] = X_train_pcs[:, i]
                    X_val_groups[col_name] = X_val_pcs[:, i]

                # Store loadings (abs value) for attribution
                loadings = pd.DataFrame(
                    np.abs(pca.components_[:n_pcs].T),
                    index=features,
                    columns=[f"PC{i+1}" for i in range(n_pcs)]
                )
                loadings_map[group_key_base] = loadings

            elif len(features) == 1:
                col_name = f"{group_key_base}_PC1"
                X_train_groups[col_name] = X_train[features[0]]
                X_val_groups[col_name] = X_val[features[0]]
                # Dummy loading
                loadings_map[group_key_base] = pd.DataFrame(
                    [1.0], index=features, columns=["PC1"]
                )

        return X_train_groups, X_val_groups, loadings_map, cluster_map

    def _compute_lgbm_importance_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Compute MDI OOF, Stability, and OOF IC with strict leak prevention.
        Handles both Classification and Regression.
        """
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        tprint_info("🚀 Starting MDI OOF & Stability Analysis...")

        cv = PurgedKFoldTime(n_splits=5)

        fold_stats = []
        oof_preds_accum = pd.Series(np.nan, index=X.index)

        # OOF Feature ICs: List of Dicts {feature: ic}
        fold_feature_ics = []

        # Determine metric and objective
        if self.is_regression:
            metric = 'rmse'
            objective = 'regression'
        else:
            metric = 'auc'
            objective = 'binary'

        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # --- Dynamic Feature Generation (Group MDI) ---
            if self.use_group_mdi:
                X_train_fold, X_val_fold, loadings, cluster_map = self._fit_transform_pca_groups(X_train, X_val)
            else:
                X_train_fold, X_val_fold = X_train, X_val
                loadings = None

            # --- Train ---
            if self.is_regression:
                model = lgb.LGBMRegressor(**self.lgbm_params)
            else:
                model = lgb.LGBMClassifier(**self.lgbm_params)

            model.fit(
                X_train_fold, y_train,
                feature_name='auto',
                categorical_feature='auto',
                eval_metric=metric
            )

            # --- Importance ---
            imp_raw = model.feature_importances_
            imp_log = np.log1p(imp_raw)

            # Normalize within fold
            if imp_log.sum() > 0:
                imp_norm = imp_log / imp_log.sum()
            else:
                imp_norm = imp_log

            # Create Series
            fold_imp_series = pd.Series(imp_norm, index=X_train_fold.columns)

            # Attribute back if Group MDI
            if self.use_group_mdi and loadings:
                fold_feat_imp = pd.Series(0.0, index=X.columns)
                for group_key, load_df in loadings.items():
                    for pc_col in load_df.columns:
                        col_name = f"{group_key}_{pc_col}"
                        if col_name in fold_imp_series:
                            val = fold_imp_series[col_name]
                            weights = load_df[pc_col] # Abs loadings
                            fold_feat_imp[weights.index] += val * weights
                fold_imp = fold_feat_imp
            else:
                fold_imp = fold_imp_series

            # --- Calculate Feature-Level OOF IC (Spearman) ---
            # Must compute on validation set only
            current_fold_ics = {}
            for col in X.columns:
                try:
                    # Get feature validation data
                    if col not in X_val.columns: continue

                    ic_val, _ = spearmanr(X_val[col], y_val)
                    if np.isnan(ic_val): ic_val = 0.0
                    current_fold_ics[col] = ic_val
                except Exception:
                    current_fold_ics[col] = 0.0
            fold_feature_ics.append(current_fold_ics)

            # --- Store Fold Stats ---
            fold_stats.append(fold_imp)

            # --- OOF Preds ---
            if self.is_regression:
                preds = model.predict(X_val_fold)
            elif hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val_fold)[:, 1]
            else:
                preds = model.predict(X_val_fold)
            oof_preds_accum.iloc[val_idx] = preds

        # --- Aggregation ---
        imp_df = pd.DataFrame(fold_stats).fillna(0.0) # Folds x Features

        # 1. Median Log Gain
        median_gain = imp_df.median()

        # 2. IQR Log Gain
        iqr_gain = imp_df.apply(iqr)

        # 3. Top-K Frequency
        k = max(1, int(len(imp_df.columns) * 0.2))
        topk_mask = imp_df.apply(lambda x: x >= x.nlargest(k).min(), axis=1)
        topk_freq = topk_mask.mean()

        # 4. Stability
        stability = median_gain / (iqr_gain + 1e-9)

        # 5. OOF IC (Model Prediction)
        valid_mask = oof_preds_accum.notna() & y.notna()
        if valid_mask.sum() > 10:
            global_oof_ic, _ = spearmanr(y[valid_mask], oof_preds_accum[valid_mask])
        else:
            global_oof_ic = 0.0

        # 6. Feature-Level OOF IC (Aggregated)
        ic_df = pd.DataFrame(fold_feature_ics).fillna(0.0)
        mean_feat_ic = ic_df.median()

        tprint_info(f"   📊 MDI OOF Stats:")
        tprint_info(f"      Median Gain: [{median_gain.min():.4f}, {median_gain.max():.4f}]")
        tprint_info(f"      Top-K Freq:  [{topk_freq.min():.2f}, {topk_freq.max():.2f}]")
        tprint_info(f"      OOF IC (Model): {global_oof_ic:.4f}")

        return {
            'mean_gain': median_gain,
            'stability': stability,
            'oof_ic': global_oof_ic,
            'feature_oof_ic': mean_feat_ic,
            'topk_freq': topk_freq,
            'median_log_gain': median_gain, # Mapping legacy names
            'iqr_log_gain': iqr_gain
        }

    def _rank_normalize(self, s: pd.Series, invert: bool = False) -> pd.Series:
        """Rank (Percentile) Normalization."""
        if s.empty: return s
        ranks = rankdata(s, method='average')
        norm = (ranks - 1) / (len(s) - 1 + 1e-9)
        norm = (ranks - 1) / (len(s) - 1 + 1e-9)
        if invert:
            return pd.Series(1.0 - norm, index=s.index)
        return pd.Series(norm, index=s.index)

    # Class-level cache
    _CACHE = {}

    def _compute_input_hash(self, X: pd.DataFrame, y: pd.Series, use_entropy_as_king: bool) -> str:
        try:
            from pandas.util import hash_pandas_object
            import hashlib
            h_X = hashlib.md5(hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
            h_y = hashlib.md5(hash_pandas_object(y, index=True).values.tobytes()).hexdigest()
            config_str = f"{self.n_estimators}_{self.max_features}_{self.gain_weight}_{self.depth_weight}_{self.ic_weight}_{self.entropy_weight}_{use_entropy_as_king}"
            config_str += f"_{self.stability_weight}_{self.use_lgbm}_{self.use_group_mdi}_{self.max_cluster_size}_{self.topk_freq_threshold}_{self.is_regression}"
            return f"{h_X}_{h_y}_{config_str}"
        except Exception:
            return None

    def run_selection(self, X: pd.DataFrame, y: pd.Series, use_entropy_as_king: bool = False) -> List[str]:
        # Detect mode if not set
        if self.is_regression is None:
            # Check target type
            if pd.api.types.is_float_dtype(y) and y.nunique() > 10:
                self.is_regression = True
                tprint_info("   ℹ️ Detected Regression target")
            else:
                self.is_regression = False
                tprint_info("   ℹ️ Detected Classification target")
        # Cache Check
        cache_key = self._compute_input_hash(X, y, use_entropy_as_king)
        if cache_key and cache_key in self._CACHE:
            tprint_success(f"⚡ [DePrado] Cache Hit! Returning pre-computed selection")
            return self._CACHE[cache_key]
        
        start_time = time.time()
        tprint_info("🚀 Starting De Prado Feature Selection Engine...")
        
        if X.empty: raise ValueError("Empty feature matrix")
        X = X.fillna(0)
        
        # 1. Global Clustering (for reporting/ExtraTrees)
        tprint_info("🔍 Step 1: Global Clustering (for reporting)...")
        cluster_map = self._get_onc_clusters(X) # Use updated method with recursion
        self.cluster_labels_ = cluster_map
        
        # Enhanced MDI
        lgbm_res = {}
        
        if self.use_lgbm:
            tprint_info("🔥 Step 1b: Running Enhanced MDI (LGBM CV)...")
            try:
                lgbm_res = self._compute_lgbm_importance_cv(X, y)
                self.lgbm_metrics_ = lgbm_res
            except Exception as e:
                tprint_error(f"❌ Enhanced MDI failed: {e}")

        # 2. Train ExtraTrees for hierarchy/fallback
        tprint_info("🌳 Step 2: Training ExtraTrees...")
        if self.is_regression:
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=False
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=False
            )
            if len(np.unique(y)) == 2: model.set_params(class_weight="balanced")
        
        X_fit, y_fit = X, y
        if len(X) > 7000:
            np.random.seed(self.random_state)
            idx = np.random.choice(len(X), 7000, replace=False)
            X_fit, y_fit = X.iloc[idx], y.iloc[idx]
        model.fit(X_fit, y_fit)
        
        # 3. Metrics
        tprint_info("📈 Step 3: Computing metrics...")
        feature_names = X.columns.tolist()
        
        # Gain
        if self.use_lgbm and 'mean_gain' in lgbm_res:
            gain = lgbm_res['mean_gain']
        else:
            mdi_metrics = self._compute_advanced_mdi(model, feature_names)
            gain = pd.Series(mdi_metrics["gain"])
        
        depth = self._get_tree_hierarchy(model, feature_names)
        
        # IC - Use OOF if available
        if self.use_lgbm and 'feature_oof_ic' in lgbm_res and lgbm_res['feature_oof_ic'] is not None:
            ic_series = lgbm_res['feature_oof_ic'] # Signed IC
            tprint_info("   ✅ Using OOF IC (Leak-Safe)")
        else:
            ic_scores = {}
            for col in X.columns:
                try:
                    ic_val, _ = spearmanr(X[col], y)
                    ic_scores[col] = ic_val if not np.isnan(ic_val) else 0.0
                except Exception:
                    ic_scores[col] = 0.0
            ic_series = pd.Series(ic_scores)
        entropy_scores = {}
        for col in X.columns:
            try:
                disc = pd.qcut(X[col], q=10, duplicates='drop')
                if disc.nunique() < 2: entropy_scores[col] = 0.0
                else: entropy_scores[col] = entropy(disc.value_counts(normalize=True))
            except Exception:
                entropy_scores[col] = 0.0
        ent_series = pd.Series(entropy_scores)
        
        # 4. Score (Rank Normalization)
        tprint_info("⚖️  Step 4: Scoring (Rank Normalized)...")
        
        score_gain = self._rank_normalize(gain)
        score_depth = self._rank_normalize(depth, invert=True)
        score_ic = self._rank_normalize(ic_series.abs()) # Score absolute IC
        score_ent = self._rank_normalize(ent_series)

        score_stability = pd.Series(0.0, index=X.columns)
        if self.use_lgbm and 'stability' in lgbm_res:
            # Combine Stability and Top-K Freq
            raw_stab = lgbm_res['stability']
            topk = lgbm_res.get('topk_freq', pd.Series(0, index=X.columns))
            # Stability score is mixture
            score_stability = 0.7 * self._rank_normalize(raw_stab) + 0.3 * self._rank_normalize(topk)
        
        composite = (
            self.gain_weight * score_gain + 
            self.stability_weight * score_stability +
            self.ic_weight * score_ic +
            self.depth_weight * score_depth + 
            self.entropy_weight * score_ent
        )
        # 5. Store
        self.feature_stats_ = pd.DataFrame({
            "Cluster": cluster_map,
            "Gain": gain,
            "IC": ic_series,
            "MeanDepth": depth,
            "Stability": lgbm_res.get('stability', 0.0),
            "TopKFreq": lgbm_res.get('topk_freq', 0.0),
            "CompositeScore": composite
        })
        
        # 6. Selection (Representative Selection)
        tprint_info(f"👑 Step 6: Selection...")
        selected_features = []
        
        # Top-K Freq Gate
        gate_mask = pd.Series(True, index=X.columns)
        if self.use_lgbm:
            topk = lgbm_res.get('topk_freq', pd.Series(0, index=X.columns))
            gate_mask = topk >= self.topk_freq_threshold
            tprint_info(f"   🚪 Gating: Dropped {(~gate_mask).sum()} features with TopKFreq < {self.topk_freq_threshold}")

            if gate_mask.sum() == 0:
                tprint_warning("   ⚠️ Gating removed all features; disabling gate for this run")
                gate_mask = pd.Series(True, index=X.columns)

        # Diversity Selection: Allow multiple reps
        for cluster_id in sorted(cluster_map.unique()):
            cluster_features = self.feature_stats_[
                (self.feature_stats_["Cluster"] == cluster_id) & gate_mask
            ]
            if len(cluster_features) == 0: continue
            
            # Rank members
            ranked = cluster_features.sort_values("CompositeScore", ascending=False)
            
            # Add Primary
            primary_feat = ranked.index[0]
            selected_features.append(primary_feat)

            # Add Secondary if cluster is high-performing and large
            # Guardrail C: Incremental value rule
            if len(ranked) > 5 and len(ranked) > 1:
                secondary_feat = ranked.index[1]
                # If secondary is within 5% score of primary
                if ranked.iloc[1]["CompositeScore"] > 0.95 * ranked.iloc[0]["CompositeScore"]:
                    # Check orthogonality
                    try:
                        corr_val = X[primary_feat].corr(X[secondary_feat])
                        if abs(corr_val) < 0.8: # Only add if sufficiently different
                            selected_features.append(secondary_feat)
                    except: pass
        
        self.selected_features_ = selected_features
        self._print_detailed_deprado_report(X)
        
        if cache_key: self._CACHE[cache_key] = selected_features
        return selected_features

    # ... (Keep getters and reporting methods) ...
    def get_feature_stats(self) -> pd.DataFrame:
        if self.feature_stats_ is None: raise ValueError("Run selection first")
        return self.feature_stats_.copy()
        
    def get_selected_features(self) -> List[str]:
        if self.selected_features_ is None: raise ValueError("Run selection first")
        return self.selected_features_.copy()

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get clustering information.

        Returns:
            Dictionary with clustering statistics
        """
        if self.cluster_labels_ is None:
            raise ValueError("Feature selection not run. Call run_selection() first.")

        cluster_counts = self.cluster_labels_.value_counts().sort_index()

        return {
            'optimal_n_clusters': self.optimal_n_clusters_,
            'silhouette_scores': self.silhouette_scores_,
            'cluster_sizes': cluster_counts.to_dict(),
            'avg_cluster_size': cluster_counts.mean(),
            'max_cluster_size': cluster_counts.max(),
            'min_cluster_size': cluster_counts.min()
        }
    def get_report(self) -> pd.DataFrame:
        if self.feature_stats_ is None: raise ValueError("Run selection first")
        return self.feature_stats_.loc[self.selected_features_].sort_values('CompositeScore', ascending=False)

    def _print_detailed_deprado_report(self, X: pd.DataFrame) -> None:
        """
        Print detailed De Prado feature selection report.
        """
        if self.feature_stats_ is None or self.selected_features_ is None:
            tprint_warning("⚠️ No feature statistics available for detailed reporting")
            return
        
        tprint_info("👑 De Prado Feature Selection Report:")
        tprint_info(f"📊 {self.optimal_n_clusters_} clusters found, {len(self.selected_features_)} features selected")
        
        for cluster_id in sorted(self.feature_stats_["Cluster"].unique()):
            cluster_features = self.feature_stats_[self.feature_stats_["Cluster"] == cluster_id]
            if len(cluster_features) == 0: continue
            
            # Find selected members
            selected_in_cluster = [f for f in cluster_features.index if f in self.selected_features_]
            
            if selected_in_cluster:
                rep_name = selected_in_cluster[0]
                rep_data = cluster_features.loc[rep_name]

                stats_parts = [f"✅ {len(selected_in_cluster)} Reps (Top: {rep_name}, Score: {rep_data['CompositeScore']:.3f})"]
                if 'Gain' in rep_data: stats_parts.append(f"Gain: {rep_data['Gain']:.4f}")
                if 'TopKFreq' in rep_data: stats_parts.append(f"TopK: {rep_data['TopKFreq']:.2f}")

                tprint_info(f"   Cluster {cluster_id}: {', '.join(stats_parts)}")


def de_prado_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 1000,
    max_clusters: int = 12,
    gain_weight: float = 0.5,
    depth_weight: float = 0.5,
    random_state: int = 42,
    # New params exposed in convenience function
    use_lgbm: bool = True,
    stability_weight: float = 0.3,
    use_group_mdi: bool = True,
    max_cluster_size: int = 30,
    topk_freq_threshold: float = 0.4,  # Updated default for robustness
    is_regression: Optional[bool] = None
) -> Tuple[pd.DataFrame, DePradoFeatureEngine]:
    """
    Convenience function for De Prado feature selection.
    """
    engine = DePradoFeatureEngine(
        n_estimators=n_estimators,
        max_clusters=max_clusters,
        gain_weight=gain_weight,
        depth_weight=depth_weight,
        random_state=random_state,
        use_lgbm=use_lgbm,
        stability_weight=stability_weight,
        use_group_mdi=use_group_mdi,
        max_cluster_size=max_cluster_size,
        topk_freq_threshold=topk_freq_threshold,
        is_regression=is_regression
    )
    
    selected_features = engine.run_selection(X, y)
    X_selected = X[selected_features].copy()
    
    return X_selected, engine
