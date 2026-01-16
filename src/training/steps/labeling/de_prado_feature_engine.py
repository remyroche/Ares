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
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import FeatureAgglomeration
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
        depth_weight: float = 0.2,
        ic_weight: float = 0.2,
        entropy_weight: float = 0.2,
        stability_weight: float = 0.0,
        min_samples_leaf: int = 30,
        max_features: str = 'log2',
        # New parameters for Enhanced MDI
        use_lgbm: bool = False,
        lgbm_params: Optional[Dict[str, Any]] = None,
        use_group_mdi: bool = False
    ):
        """
        Initialize De Prado Feature Engine.
        
        Args:
            n_estimators: Number of trees in ExtraTrees
            max_clusters: Maximum number of clusters to consider
            min_cluster_size: Minimum samples per cluster
            random_state: Random seed for reproducibility
            gain_weight: Weight for gain in composite score (default: 0.4)
            depth_weight: Weight for depth proximity in composite score (default: 0.2)
            ic_weight: Weight for Information Coefficient in composite score (default: 0.2)
            entropy_weight: Weight for Shannon Entropy in composite score (default: 0.2)
            stability_weight: Weight for Stability Score (MDI OOF stability)
            min_samples_leaf: Minimum samples per leaf in ExtraTrees
            max_features: Max features considered for each split
            use_lgbm: Use LightGBM for MDI/Stability calculation
            lgbm_params: Parameters for LightGBM
            use_group_mdi: Use Group-Aware MDI (PCA-based)
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
            # Update defaults with provided params
            for k, v in defaults.items():
                if k not in self.lgbm_params:
                    self.lgbm_params[k] = v

        # Results storage
        self.feature_stats_ = None
        self.selected_features_ = None
        self.cluster_labels_ = None
        self.optimal_n_clusters_ = None
        self.silhouette_scores_ = None
        self.lgbm_importances_ = None
        
    def _get_onc_clusters(self, X: pd.DataFrame) -> pd.Series:
        """
        Finds optimal feature clusters using Multi-criteria ONC.
        """
        tprint_info("🔍 Finding optimal feature clusters (Multi-criteria ONC)...")
        onc_start = time.time()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Compute correlation matrix
        corr = X.corr().fillna(0)
        corr = np.clip(corr, -1, 1)

        # Representation where each feature becomes a "sample" vector for metrics
        feature_sample_matrix = X.T
        
        # Convert to correlation distance
        dist = 1 - np.abs(corr)
        np.fill_diagonal(dist.values, 0)
        dist = np.maximum(dist, 0)
        
        best_k, best_composite_score = 2, -1
        scores_history = {}
        max_k = min(self.max_clusters, max(2, len(X.columns) // 2))
        
        for k in range(2, max_k + 1):
            try:
                clusterer = FeatureAgglomeration(n_clusters=k, linkage='average')
                clusterer.fit(X)
                cluster_labels = clusterer.labels_
                
                if len(np.unique(cluster_labels)) > 1:
                    cv_ratio = self._calculate_cv_ratio(feature_sample_matrix, cluster_labels)
                    dbi = davies_bouldin_score(feature_sample_matrix, cluster_labels)
                    silhouette = silhouette_score(dist, cluster_labels, metric='precomputed')
                    ch = calinski_harabasz_score(feature_sample_matrix, cluster_labels)
                    
                    cv_score = cv_ratio
                    dbi_score = 1.0 / (1.0 + dbi)
                    silhouette_score_norm = (silhouette + 1.0) / 2.0
                    
                    scores_history[k] = {
                        'cv_ratio': cv_ratio,
                        'dbi': dbi,
                        'silhouette': silhouette,
                        'ch': ch,
                        'cv_score': cv_score,
                        'dbi_score': dbi_score,
                        'silhouette_score_norm': silhouette_score_norm
                    }
                    
                    composite_score = (
                        0.50 * cv_score +
                        0.30 * dbi_score +
                        0.20 * silhouette_score_norm
                    )
                    scores_history[k]['composite'] = composite_score
                    
                    if composite_score > best_composite_score:
                        best_composite_score, best_k = composite_score, k
                        
            except Exception:
                continue

        # Final clustering
        if best_k == 1:
            final_labels = np.zeros(len(X.columns))
        else:
            final_clusterer = FeatureAgglomeration(n_clusters=best_k, linkage='average')
            final_clusterer.fit(X.T)
            final_labels = final_clusterer.labels_
            
            # Fallback if label mismatch
            if len(final_labels) != len(X.columns):
                 final_labels = np.arange(len(X.columns)) % best_k
        
        self.optimal_n_clusters_ = best_k
        self.silhouette_scores_ = scores_history

        return pd.Series(final_labels, index=X.columns)
    
    def _calculate_cv_ratio(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate CV Ratio."""
        try:
            if X.shape[1] > 5000:
                np.random.seed(42)
                sample_cols = np.random.choice(X.shape[1], 5000, replace=False)
                X = X.iloc[:, sample_cols]
            
            overall_centroid = X.mean(axis=0)
            bcss = 0.0
            wcss = 0.0
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = X.loc[cluster_mask] if isinstance(X, pd.DataFrame) else X[cluster_mask, :]
                
                if len(cluster_data) > 0:
                    cluster_centroid = cluster_data.mean(axis=0)
                    if isinstance(cluster_data, pd.DataFrame):
                        wcss += ((cluster_data - cluster_centroid) ** 2).sum().sum()
                    else:
                        wcss += ((cluster_data - cluster_centroid) ** 2).sum()
                    
                    n_cluster = len(cluster_data)
                    if isinstance(cluster_centroid, pd.Series):
                        bcss += n_cluster * ((cluster_centroid - overall_centroid) ** 2).sum()
                    else:
                        bcss += n_cluster * np.sum((cluster_centroid - overall_centroid) ** 2)
            
            if wcss > 0:
                cv_ratio = bcss / wcss
                cv_ratio = min(cv_ratio / 10.0, 1.0)
            else:
                cv_ratio = 0.0
                
            return cv_ratio
            
        except Exception:
            return 0.0
    
    def _get_tree_hierarchy(self, model: ExtraTreesClassifier, feature_names: List[str]) -> pd.Series:
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
                mean_depths[name] = np.mean(depth_list)
            else:
                mean_depths[name] = max_depth_overall
        
        return pd.Series(mean_depths)
    
    def _compute_advanced_mdi(self, model: ExtraTreesClassifier, feature_names: List[str]) -> Dict[str, float]:
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

    def _get_pca_groups(self, X: pd.DataFrame, cluster_labels: pd.Series) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Construct PCA-based group features.
        Returns:
            X_groups: DataFrame of PCs
            loadings_map: Dict mapping group_key -> loadings DataFrame
        """
        X_groups = pd.DataFrame(index=X.index)
        loadings_map = {}

        for cluster_id in sorted(cluster_labels.unique()):
            features = cluster_labels[cluster_labels == cluster_id].index
            group_key_base = f"Cluster_{cluster_id}"

            if len(features) > 1:
                # Retain up to 3 PCs or 70% variance
                pca = PCA(n_components=min(3, len(features)))
                scaler = StandardScaler()
                X_cluster = scaler.fit_transform(X[features])
                pcs = pca.fit_transform(X_cluster)

                # Check explained variance
                expl_var = np.cumsum(pca.explained_variance_ratio_)
                n_pcs = np.searchsorted(expl_var, 0.70) + 1
                n_pcs = min(n_pcs, pcs.shape[1])

                # Store PCs
                for i in range(n_pcs):
                    col_name = f"{group_key_base}_PC{i+1}"
                    X_groups[col_name] = pcs[:, i]

                # Store loadings (abs value) for attribution
                loadings = pd.DataFrame(
                    np.abs(pca.components_[:n_pcs].T),
                    index=features,
                    columns=[f"PC{i+1}" for i in range(n_pcs)]
                )
                loadings_map[group_key_base] = loadings

            elif len(features) == 1:
                col_name = f"{group_key_base}_PC1"
                X_groups[col_name] = X[features[0]]
                # Dummy loading
                loadings_map[group_key_base] = pd.DataFrame(
                    [1.0], index=features, columns=["PC1"]
                )

        return X_groups, loadings_map

    def _compute_lgbm_importance_cv(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MDI OOF, Stability, and OOF IC.
        Using Robust Signal-to-Variation (Median / IQR).
        """
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        tprint_info("🚀 Starting MDI OOF & Stability Analysis...")

        cv = PurgedKFoldTime(n_splits=5)

        importances = []
        feature_names = X.columns
        oof_preds = pd.Series(np.nan, index=X.index)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            model = lgb.LGBMClassifier(**self.lgbm_params)

            model.fit(
                X_train, y_train,
                feature_name='auto',
                categorical_feature='auto',
                eval_metric='auc'
            )

            # 1. Feature Importance (Log1p transformed)
            imp = model.feature_importances_

            # Explicitly handle missing features (zero gain)
            # (LGBM returns importance for all features it saw in init,
            # so if X columns match, we are good. Zero splits = zero gain.)

            # Log transform for robustness
            imp_log = np.log1p(imp)

            # Normalize to sum to 1 (relative importance within fold)
            if imp_log.sum() > 0:
                imp_log = imp_log / imp_log.sum()

            importances.append(imp_log)

            # 2. OOF Predictions
            if hasattr(model, "predict_proba"):
                oof_preds.iloc[val_idx] = model.predict_proba(X_val)[:, 1]
            else:
                oof_preds.iloc[val_idx] = model.predict(X_val)

        # Stack importances
        imp_df = pd.DataFrame(importances, columns=feature_names)

        # Robust Aggregation
        median_gain = imp_df.median()
        iqr_gain = imp_df.apply(iqr)

        # Robust Stability = Median / (IQR + eps)
        stability = median_gain / (iqr_gain + 1e-9)

        # OOF IC Calculation (Spearman)
        # Align y and oof_preds (drop NaNs from purge/embargo)
        valid_mask = oof_preds.notna() & y.notna()
        if valid_mask.sum() > 10:
            oof_ic, _ = spearmanr(y[valid_mask], oof_preds[valid_mask])
        else:
            oof_ic = 0.0

        tprint_info(f"   📊 MDI OOF: Median Gain range [{median_gain.min():.4f}, {median_gain.max():.4f}]")
        tprint_info(f"   📊 Stability: Score range [{stability.min():.4f}, {stability.max():.4f}]")
        tprint_info(f"   📊 OOF IC: {oof_ic:.4f}")

        return median_gain, stability, oof_ic

    def _rank_normalize(self, s: pd.Series, invert: bool = False) -> pd.Series:
        """Rank (Percentile) Normalization."""
        if s.empty: return s
        ranks = rankdata(s, method='average')
        # Normalize to [0, 1]
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
            config_str += f"_{self.stability_weight}_{self.use_lgbm}_{self.use_group_mdi}"
            return f"{h_X}_{h_y}_{config_str}"
        except Exception:
            return None

    def run_selection(self, X: pd.DataFrame, y: pd.Series, use_entropy_as_king: bool = False) -> List[str]:
        # Cache Check
        cache_key = self._compute_input_hash(X, y, use_entropy_as_king)
        if cache_key and cache_key in self._CACHE:
            tprint_success(f"⚡ [DePrado] Cache Hit! Returning pre-computed selection")
            return self._CACHE[cache_key]
        
        start_time = time.time()
        tprint_info("🚀 Starting De Prado Feature Selection Engine...")
        
        if X.empty: raise ValueError("Empty feature matrix")
        X = X.fillna(0)
        
        # 1. Cluster Features
        tprint_info("🔍 Step 1: Finding optimal feature clusters (ONC)...")
        cluster_map = self._get_onc_clusters(X)
        self.cluster_labels_ = cluster_map
        
        # Enhanced MDI
        lgbm_gain = None
        lgbm_stability = None
        oof_ic = 0.0
        
        if self.use_lgbm:
            tprint_info("🔥 Step 1b: Running Enhanced MDI (LGBM CV)...")
            try:
                if self.use_group_mdi:
                    tprint_info("   🧩 Using Group-Aware MDI (PCA Groups)")
                    X_input, loadings_map = self._get_pca_groups(X, cluster_map)

                    mean_gain, stability, oof_ic = self._compute_lgbm_importance_cv(X_input, y)

                    lgbm_gain = pd.Series(index=X.columns, dtype=float)
                    lgbm_stability = pd.Series(index=X.columns, dtype=float)

                    # Attribute back to features
                    for cluster_id in cluster_map.unique():
                        features = cluster_map[cluster_map == cluster_id].index
                        group_key_base = f"Cluster_{cluster_id}"

                        if group_key_base not in loadings_map:
                            continue

                        loadings = loadings_map[group_key_base] # (n_features, n_pcs)

                        # Find relevant PCs for this group (columns of loadings)
                        pcs = loadings.columns

                        # Calculate weighted stability/gain for each feature based on loading
                        # Gain_feat = Sum(Gain_PC * AbsLoading_PC)
                        feat_gain_accum = np.zeros(len(features))
                        feat_stab_accum = np.zeros(len(features))

                        for pc in pcs:
                            col_name = f"{group_key_base}_{pc}"
                            if col_name in mean_gain:
                                g = mean_gain[col_name]
                                s = stability[col_name]
                                w = loadings[pc].values # Abs loadings

                                feat_gain_accum += g * w
                                feat_stab_accum += s * w

                        lgbm_gain[features] = feat_gain_accum
                        lgbm_stability[features] = feat_stab_accum

                else:
                    lgbm_gain, lgbm_stability, oof_ic = self._compute_lgbm_importance_cv(X, y)

            except Exception as e:
                tprint_error(f"❌ Enhanced MDI failed: {e}")

        # 2. Train ExtraTrees for hierarchy/fallback
        tprint_info("🌳 Step 2: Training ExtraTrees...")
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
        
        if self.use_lgbm and lgbm_gain is not None:
            gain = lgbm_gain
        else:
            mdi_metrics = self._compute_advanced_mdi(model, feature_names)
            gain = pd.Series(mdi_metrics["gain"])
        
        depth = self._get_tree_hierarchy(model, feature_names)
        
        ic_scores = {}
        for col in X.columns:
            try:
                ic_val, _ = spearmanr(X[col], y)
                ic_scores[col] = abs(ic_val) if not np.isnan(ic_val) else 0.0
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
        score_ic = self._rank_normalize(ic_series)
        score_ent = self._rank_normalize(ent_series)
        
        score_stability = pd.Series(0.0, index=X.columns)
        if self.use_lgbm and lgbm_stability is not None:
            score_stability = self._rank_normalize(lgbm_stability)
        
        composite = (
            self.gain_weight * score_gain + 
            self.depth_weight * score_depth + 
            self.ic_weight * score_ic + 
            self.entropy_weight * score_ent +
            self.stability_weight * score_stability
        )
        
        # If OOF IC is high, boost composite? (Optional, kept simple for now)
        
        # 5. Store
        self.feature_stats_ = pd.DataFrame({
            "Cluster": cluster_map,
            "Gain": gain,
            "IC": ic_series,
            "MeanDepth": depth,
            "Stability": lgbm_stability if lgbm_stability is not None else 0.0,
            "CompositeScore": composite
        })
        
        # 6. Selection (King)
        tprint_info(f"👑 Step 6: Selection...")
        selected_features = []
        
        for cluster_id in sorted(cluster_map.unique()):
            cluster_features = self.feature_stats_[self.feature_stats_["Cluster"] == cluster_id]
            if len(cluster_features) == 0: continue
            
            # Select "King" based on Composite
            if use_entropy_as_king:
                 best_feature = cluster_features.loc[cluster_features["EntropyScore"].idxmax()]
            else:
                 best_feature = cluster_features.loc[cluster_features["CompositeScore"].idxmax()]
            
            selected_features.append(best_feature.name)
        
        self.selected_features_ = selected_features
        self._print_detailed_deprado_report(X)
        
        if cache_key: self._CACHE[cache_key] = selected_features
        return selected_features

    # ... (Keep getters and reporting methods same as before, ensuring they use new stats) ...
    def get_feature_stats(self) -> pd.DataFrame:
        if self.feature_stats_ is None: raise ValueError("Run selection first")
        return self.feature_stats_.copy()
        
    def get_selected_features(self) -> List[str]:
        if self.selected_features_ is None: raise ValueError("Run selection first")
        return self.selected_features_.copy()

    def get_report(self) -> pd.DataFrame:
        if self.feature_stats_ is None: raise ValueError("Run selection first")
        return self.feature_stats_.loc[self.selected_features_].sort_values('CompositeScore', ascending=False)

    def _print_detailed_deprado_report(self, X: pd.DataFrame) -> None:
        """
        Print detailed De Prado feature selection report with cluster-by-cluster analysis.
        
        Args:
            X: Original feature matrix
        """
        if self.feature_stats_ is None or self.selected_features_ is None:
            tprint_warning("⚠️ No feature statistics available for detailed reporting")
            return
        
        max_display = 50  # Limit output for large feature sets
        
        tprint_info("👑 De Prado Feature Selection Report:")
        tprint_info(f"📊 {self.optimal_n_clusters_} clusters found, {len(self.selected_features_)} king features selected")
        
        # Cluster-by-cluster analysis
        for cluster_id in sorted(self.feature_stats_["Cluster"].unique()):
            cluster_features = self.feature_stats_[self.feature_stats_["Cluster"] == cluster_id]
            cluster_size = len(cluster_features)
            
            if cluster_size == 0:
                continue
            
            # Find king feature
            king_feature = cluster_features.loc[cluster_features["CompositeScore"].idxmax()]
            is_king_selected = king_feature.name in self.selected_features_
            
            tprint_info(f"🔍 Cluster {cluster_id} ({cluster_size} features):")
            
            if is_king_selected:
                stats_parts = [f"👑 KING: {king_feature.name} (score: {king_feature['CompositeScore']:.3f}) ✅ SELECTED"]
                if 'Gain' in king_feature.index:
                    stats_parts.append(f"Gain: {king_feature['Gain']:.4f}")
                if 'Cover' in king_feature.index:
                    stats_parts.append(f"Cover: {king_feature['Cover']:.4f}")
                if 'MeanDepth' in king_feature.index:
                    stats_parts.append(f"Depth: {king_feature['MeanDepth']:.2f}")
                if 'Stability' in king_feature.index and self.use_lgbm:
                    stats_parts.append(f"Stab: {king_feature['Stability']:.2f}")
                tprint_info(f"      📊 {', '.join(stats_parts)}")
            else:
                tprint_info(f"   ❌ No king selected from cluster {cluster_id}")
            
            # Show other features in cluster (discarded)
            other_features = cluster_features[cluster_features.index != king_feature.name]
            if len(other_features) > 0:
                tprint_info(f"   ❌ Discarded features ({len(other_features)}):")
                
                # Sort by composite score to show best alternatives
                sorted_others = other_features.sort_values("CompositeScore", ascending=False)
                for i, (feature_name, feature_data) in enumerate(sorted_others.head(max_display).iterrows()):
                    score_diff = king_feature["CompositeScore"] - feature_data["CompositeScore"]
                    reason = f"lost to king by {score_diff:.3f} points"
                    tprint_info(f"      ❌ {feature_name}: {feature_data['CompositeScore']:.3f} ({reason})")
                
                if len(other_features) > max_display:
                    tprint_info(f"      ... and {len(other_features) - max_display} more discarded features")
        
        # Overall statistics
        tprint_info(f"📊 Selection Summary:")
        tprint_info(f"   📈 Input features: {len(X.columns)}")
        tprint_info(f"   🎯 Clusters formed: {self.optimal_n_clusters_}")
        tprint_info(f"   👑 Kings selected: {len(self.selected_features_)}")
        tprint_info(f"   ❌ Features discarded: {len(X.columns) - len(self.selected_features_)}")
        tprint_info(f"   📉 Reduction ratio: {(1 - len(self.selected_features_)/len(X.columns)):.1%}")
        
        # Quality metrics
        if len(self.selected_features_) > 0:
            selected_stats = self.feature_stats_[self.feature_stats_.index.isin(self.selected_features_)]
            discarded_stats = self.feature_stats_[~self.feature_stats_.index.isin(self.selected_features_)]
            
            tprint_info(f"📊 Quality Comparison:")
            tprint_info(f"   📈 Selected features avg composite score: {selected_stats['CompositeScore'].mean():.3f}")
            tprint_info(f"   📉 Discarded features avg composite score: {discarded_stats['CompositeScore'].mean():.3f}")
            
            if discarded_stats['CompositeScore'].mean() > 0:
                improvement = ((selected_stats['CompositeScore'].mean() - discarded_stats['CompositeScore'].mean()) / discarded_stats['CompositeScore'].mean() * 100)
                tprint_info(f"   🎯 Quality improvement: {improvement:.1f}% higher score in selected features")
        
        # Top king features
        if len(self.selected_features_) > 0:
            selected_stats = self.feature_stats_[self.feature_stats_.index.isin(self.selected_features_)]
            top_kings = selected_stats.sort_values("CompositeScore", ascending=False).head(3)
            
            tprint_info(f"🏆 Top 3 King Features:")
            for i, (feature_name, feature_data) in enumerate(top_kings.iterrows()):
                tprint_info(f"   {i+1}. {feature_name}: {feature_data['CompositeScore']:.3f} (Cluster {feature_data['Cluster']})")
        
        # Cluster size distribution
        cluster_sizes = self.feature_stats_["Cluster"].value_counts().sort_index()
        tprint_info(f"📊 Cluster Size Distribution:")
        for cluster_id, size in cluster_sizes.items():
            king_in_cluster = self.feature_stats_[
                (self.feature_stats_["Cluster"] == cluster_id) & 
                (self.feature_stats_.index.isin(self.selected_features_))
            ]
            king_name = king_in_cluster.index[0] if len(king_in_cluster) > 0 else "None"
            tprint_info(f"   Cluster {cluster_id}: {size} features → King: {king_name}")


def de_prado_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 1000,
    max_clusters: int = 12,
    gain_weight: float = 0.5,
    depth_weight: float = 0.5,
    random_state: int = 42,
    # New params exposed in convenience function
    use_lgbm: bool = False,
    stability_weight: float = 0.0,
    use_group_mdi: bool = False
) -> Tuple[pd.DataFrame, DePradoFeatureEngine]:
    """
    Convenience function for De Prado feature selection.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of trees in ExtraTrees
        max_clusters: Maximum number of clusters
        gain_weight: Weight for gain in composite score
        depth_weight: Weight for depth proximity in composite score
        random_state: Random seed
        use_lgbm: Use LightGBM for MDI/Stability
        stability_weight: Weight for Stability Score
        use_group_mdi: Use Group-Aware MDI (PCA-based)
        
    Returns:
        Tuple of (selected_features_df, fitted_engine)
    """
    engine = DePradoFeatureEngine(
        n_estimators=n_estimators,
        max_clusters=max_clusters,
        gain_weight=gain_weight,
        depth_weight=depth_weight,
        random_state=random_state,
        use_lgbm=use_lgbm,
        stability_weight=stability_weight,
        use_group_mdi=use_group_mdi
    )
    
    selected_features = engine.run_selection(X, y)
    X_selected = X[selected_features].copy()
    
    return X_selected, engine
