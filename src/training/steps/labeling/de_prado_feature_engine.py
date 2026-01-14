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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import norm, entropy, spearmanr
import warnings
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

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
        n_estimators: int = 1000,
        max_clusters: int = 12,
        min_cluster_size: int = 2,
        random_state: int = 42,
        gain_weight: float = 0.4,
        depth_weight: float = 0.2,
        ic_weight: float = 0.2,
        entropy_weight: float = 0.2,
        min_samples_leaf: int = 30,
        max_features: str = 'log2'
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
            min_samples_leaf: Minimum samples per leaf in ExtraTrees
            max_features: Max features considered for each split
        """
        self.n_estimators = n_estimators
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.gain_weight = gain_weight
        self.depth_weight = depth_weight
        self.ic_weight = ic_weight
        self.entropy_weight = entropy_weight
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        # Results storage
        self.feature_stats_ = None
        self.selected_features_ = None
        self.cluster_labels_ = None
        self.optimal_n_clusters_ = None
        self.silhouette_scores_ = None
        
    def _get_onc_clusters(self, X: pd.DataFrame) -> pd.Series:
        """
        Finds optimal feature clusters using Multi-criteria ONC.
        
        Primary: CV Ratio (BCSS/WCSS) - measures cluster separation vs cohesion
        Secondary: Davies-Bouldin Index - lower is better
        Tertiary: Silhouette Score - higher is better
        
        Args:
            X: Feature matrix
            
        Returns:
            Series of cluster labels indexed by feature names
        """
        tprint_info("🔍 Finding optimal feature clusters (Multi-criteria ONC)...")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Compute correlation matrix
        corr = X.corr().fillna(0)
        
        # Representation where each feature becomes a "sample" vector for metrics
        feature_sample_matrix = X.T
        
        # Debug: Check correlation matrix properties
        tprint_info(f"🔍 Correlation matrix shape: {corr.shape}")
        tprint_info(f"   Correlation stats: min={corr.min().min():.3f}, max={corr.max().max():.3f}")
        
        # Check for highly correlated features
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        if high_corr_pairs:
            tprint_warning(f"   Found {len(high_corr_pairs)} highly correlated pairs (>0.95)")
            for pair in high_corr_pairs[:3]:  # Show first 3
                tprint_warning(f"      {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        # Handle perfect correlation
        if corr.isin([1.0]).all().all():
            tprint_warning("All features are perfectly correlated. Using single cluster.")
            return pd.Series(0, index=X.columns)
        
        # Convert to correlation distance
        dist = 1 - np.abs(corr)  # Use absolute correlation for distance
        
        # Ensure diagonal is exactly zero (fix for the error)
        np.fill_diagonal(dist.values, 0)
        
        # Find optimal number of clusters using multi-criteria scoring
        best_k, best_composite_score = 2, -1
        scores_history = {}
        
        # Test cluster sizes from 2 to max_clusters or n_features//2
        max_k = min(self.max_clusters, max(2, len(X.columns) // 2))
        
        tprint_info(f"   🔄 Testing cluster counts K=2 to {max_k} (Multi-criteria)...")
        tprint_info(f"   📊 Metrics: CV Ratio (primary), DBI (secondary), Silhouette (tertiary)")

        for k in range(2, max_k + 1):
            try:
                clusterer = FeatureAgglomeration(n_clusters=k, linkage='average')
                # Fit directly on (n_samples, n_features); FeatureAgglomeration clusters columns
                clusterer.fit(X)
                cluster_labels = clusterer.labels_  # This should have length = n_features (50)
                
                # Debug: Check if clustering actually produced k clusters
                unique_labels = np.unique(cluster_labels)
                if len(unique_labels) != k:
                    tprint_warning(f"      - K={k}: Expected {k} clusters, got {len(unique_labels)}")
                
                # Debug: Check label length
                if len(cluster_labels) != len(X.columns):
                    tprint_warning(f"      - K={k}: Label length mismatch: {len(cluster_labels)} vs {len(X.columns)}")
                    continue
                
                # Compute multiple clustering quality metrics
                if len(unique_labels) > 1:
                    # 1. CV Ratio (BCSS/WCSS) - Primary metric
                    # Use feature_sample_matrix (features as samples)
                    cv_ratio = self._calculate_cv_ratio(feature_sample_matrix, cluster_labels)
                    
                    # 2. Davies-Bouldin Index - Secondary metric (lower is better)
                    dbi = davies_bouldin_score(feature_sample_matrix, cluster_labels)
                    
                    # 3. Silhouette Score - Tertiary metric (higher is better)
                    # Use transposed data with correlation distance
                    silhouette = silhouette_score(dist, cluster_labels, metric='precomputed')
                    
                    # 4. Calinski-Harabasz Index - Additional metric (higher is better)
                    ch = calinski_harabasz_score(feature_sample_matrix, cluster_labels)
                    
                    # Normalize each metric for composite scoring
                    # CV Ratio: higher is better (already normalized 0-1)
                    cv_score = cv_ratio
                    
                    # DBI: lower is better, normalize to 0-1 (invert)
                    dbi_score = 1.0 / (1.0 + dbi)
                    
                    # Silhouette: higher is better (already -1 to 1, shift to 0-1)
                    silhouette_score_norm = (silhouette + 1.0) / 2.0
                    
                    # CH: higher is better, normalize across all k values
                    ch_score = ch  # Will be normalized later
                    
                    # Store all scores
                    scores_history[k] = {
                        'cv_ratio': cv_ratio,
                        'dbi': dbi,
                        'silhouette': silhouette,
                        'ch': ch,
                        'cv_score': cv_score,
                        'dbi_score': dbi_score,
                        'silhouette_score_norm': silhouette_score_norm,
                        'ch_score': ch_score
                    }
                    
                    # Composite score with CV Ratio as primary (50%), DBI as secondary (30%), Silhouette as tertiary (20%)
                    composite_score = (
                        0.50 * cv_score +      # CV Ratio - Primary
                        0.30 * dbi_score +    # DBI - Secondary  
                        0.20 * silhouette_score_norm  # Silhouette - Tertiary
                    )
                    
                    scores_history[k]['composite'] = composite_score
                    
                    tprint_info(f"      - K={k}: CV={cv_ratio:.3f}, DBI={dbi_avg:.3f}, Sil={silhouette:.3f}, Comp={composite_score:.3f}")
                    
                    if composite_score > best_composite_score:
                        best_composite_score, best_k = composite_score, k
                        
                else:
                    tprint_warning(f"      - K={k}: Only 1 cluster found")
                        
            except Exception as e:
                tprint_warning(f"      - K={k}: Failed - {e}")
                continue
        
        # Normalize CH scores across all k values
        if scores_history:
            ch_values = [scores_history[k]['ch'] for k in scores_history.keys()]
            if ch_values and max(ch_values) > 0:
                ch_min, ch_max = min(ch_values), max(ch_values)
                ch_range = ch_max - ch_min
                if ch_range > 0:
                    for k in scores_history:
                        scores_history[k]['ch_score'] = (scores_history[k]['ch'] - ch_min) / ch_range
                        # Recompute composite score
                        scores_history[k]['composite'] = (
                            0.50 * scores_history[k]['cv_score'] +
                            0.30 * scores_history[k]['dbi_score'] +
                            0.20 * scores_history[k]['silhouette_score_norm']
                        )
        
        # Quality check on best solution
        if best_k == 2 and best_composite_score < 0.3 and scores_history:
            # Force more clusters if quality is too low
            min_clusters = min(3, max(2, len(X.columns) // 2))
            tprint_warning(f"   ⚠️ ONC Quality Check: Best K={best_k} has composite score {best_composite_score:.3f}")
            tprint_warning(f"   Forcing {min_clusters} clusters for diversity.")
            best_k = min_clusters
        
        # Final clustering with optimal k
        if best_k == 1:
            final_labels = np.zeros(len(X.columns))
        else:
            final_clusterer = FeatureAgglomeration(n_clusters=best_k, linkage='average')
            final_clusterer.fit(X.T)  # Transpose to cluster features
            final_labels = final_clusterer.labels_
            
            # Debug: Check final clustering
            if len(final_labels) != len(X.columns):
                tprint_warning(f"   ⚠️ Final clustering label length mismatch: {len(final_labels)} vs {len(X.columns)}")
                # Fallback: assign each feature to a cluster based on modulo
                final_labels = np.arange(len(X.columns)) % best_k
                tprint_warning(f"   🔄 Using fallback clustering assignment")
        
        self.optimal_n_clusters_ = best_k
        self.silhouette_scores_ = scores_history
        
        # Enhanced logging of results
        best_metrics = scores_history.get(best_k, {})
        tprint_success(f"✅ ONC: {best_k} optimal clusters found")
        tprint_info(f"   📊 Best metrics: CV={best_metrics.get('cv_ratio', 0):.3f}, DBI={best_metrics.get('dbi', 0):.3f}, Sil={best_metrics.get('silhouette', 0):.3f}")
        tprint_info(f"   🎯 Composite score: {best_composite_score:.3f}")
        
        # Log final cluster distribution
        try:
            final_dist = pd.Series(final_labels).value_counts().sort_index()
            tprint_info(f"   📊 Final cluster sizes: {final_dist.to_dict()}")
        except:
            pass

        return pd.Series(final_labels, index=X.columns)
    
    def _calculate_cv_ratio(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """
        Calculate CV Ratio (BCSS/WCSS) - measures cluster separation vs cohesion.
        Higher values indicate better clustering.
        
        Args:
            X: Feature matrix (transposed - features as samples)
            labels: Cluster labels for features
            
        Returns:
            CV Ratio value
        """
        try:
            # X should be features as samples, labels correspond to features
            # Calculate between-cluster sum of squares (BCSS)
            overall_centroid = X.mean(axis=0)
            bcss = 0.0
            
            # Calculate within-cluster sum of squares (WCSS)
            wcss = 0.0
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = X.loc[cluster_mask] if isinstance(X, pd.DataFrame) else X[cluster_mask, :]
                
                if len(cluster_data) > 0:
                    # Within-cluster sum of squares
                    cluster_centroid = cluster_data.mean(axis=0)
                    if isinstance(cluster_data, pd.DataFrame):
                        wcss += ((cluster_data - cluster_centroid) ** 2).sum().sum()
                    else:
                        wcss += ((cluster_data - cluster_centroid) ** 2).sum()
                    
                    # Between-cluster sum of squares
                    n_cluster = len(cluster_data)
                    if isinstance(cluster_centroid, pd.Series):
                        bcss += n_cluster * ((cluster_centroid - overall_centroid) ** 2).sum()
                    else:
                        bcss += n_cluster * np.sum((cluster_centroid - overall_centroid) ** 2)
            
            # CV Ratio (higher is better)
            if wcss > 0:
                cv_ratio = bcss / wcss
                # Normalize to roughly 0-1 range for typical financial data
                cv_ratio = min(cv_ratio / 10.0, 1.0)  # Cap at 1.0
            else:
                cv_ratio = 0.0
                
            return cv_ratio
            
        except Exception as e:
            tprint_warning(f"   ⚠️ CV Ratio calculation failed: {e}")
            return 0.0
    
    def _get_tree_hierarchy(self, model: ExtraTreesClassifier, feature_names: List[str]) -> pd.Series:
        """
        Calculates Mean First Split Depth for each feature.
        
        Args:
            model: Trained ExtraTrees model
            feature_names: List of feature names
            
        Returns:
            Series of mean depths indexed by feature names
        """
        depths = {name: [] for name in feature_names}
        max_depth_overall = 0
        
        for tree in model.estimators_:
            t = tree.tree_
            tree_depth = t.max_depth
            max_depth_overall = max(max_depth_overall, tree_depth)
            
            first_occurrence = {}
            
            def walk_node(node: int, current_depth: int):
                """Walk tree to find first occurrence of each feature."""
                if t.feature[node] != -2:  # Not a leaf node
                    feature_idx = t.feature[node]
                    if feature_idx not in first_occurrence:
                        first_occurrence[feature_idx] = current_depth
                    
                    # Recurse to children
                    walk_node(t.children_left[node], current_depth + 1)
                    walk_node(t.children_right[node], current_depth + 1)
            
            walk_node(0, 0)
            
            # Record depths
            for idx, depth in first_occurrence.items():
                if idx < len(feature_names):
                    depths[feature_names[idx]].append(depth)
        
        # Convert to mean depths, penalizing features that never appear
        mean_depths = {}
        for name, depth_list in depths.items():
            if depth_list:
                mean_depths[name] = np.mean(depth_list)
            else:
                # Features that never appear get max depth penalty
                mean_depths[name] = max_depth_overall
        
        return pd.Series(mean_depths)
    
    def _compute_advanced_mdi(self, model: ExtraTreesClassifier, feature_names: List[str]) -> Dict[str, float]:
        """
        Compute Advanced MDI metrics including Gain and Cover.
        
        Args:
            model: Trained ExtraTrees model
            feature_names: List of feature names
            
        Returns:
            Dictionary with MDI metrics
        """
        # Standard feature importances (Gain)
        gain_importances = model.feature_importances_
        
        # Compute Cover (how many samples each feature affects)
        cover_counts = np.zeros(len(feature_names))
        total_samples = 0
        
        for tree in model.estimators_:
            t = tree.tree_
            n_samples = t.n_node_samples
            
            for i in range(t.node_count):
                if t.feature[i] != -2:  # Not a leaf
                    feature_idx = t.feature[i]
                    if feature_idx < len(feature_names):
                        # Weight by number of samples passing through this node
                        cover_counts[feature_idx] += n_samples[i]
            
            total_samples += n_samples[0]  # Root node samples
        
        # Normalize cover
        cover_importances = cover_counts / (total_samples + EPS)
        
        return {
            'gain': dict(zip(feature_names, gain_importances)),
            'cover': dict(zip(feature_names, cover_importances))
        }
    
    # Class-level cache to prevent re-running selection on identical data
    _CACHE = {}

    def _compute_input_hash(self, X: pd.DataFrame, y: pd.Series, use_entropy_as_king: bool) -> str:
        """Compute a hash of the inputs for caching."""
        try:
            from pandas.util import hash_pandas_object
            import hashlib
            
            # Hash data content (values only to be fast, or index+values for safety)
            # Use columns + shape + sample of values for speed if X is huge? 
            # Ideally hash_pandas_object is safe.
            h_X = hashlib.md5(hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
            h_y = hashlib.md5(hash_pandas_object(y, index=True).values.tobytes()).hexdigest()
            
            # Hash config
            config_str = f"{self.n_estimators}_{self.max_features}_{self.gain_weight}_{self.depth_weight}_{self.ic_weight}_{self.entropy_weight}_{use_entropy_as_king}"
            
            return f"{h_X}_{h_y}_{config_str}"
        except Exception:
            return None

    def run_selection(self, X: pd.DataFrame, y: pd.Series, use_entropy_as_king: bool = False) -> List[str]:
        """
        Run complete De Prado feature selection pipeline.
        With caching support.
        
        Args:
            X: Feature matrix
            y: Target labels
            use_entropy_as_king: If True, entropy is the sole criterion for intra-cluster selection
            
        Returns:
            List of selected feature names
        """
        # --- CACHE CHECK ---
        cache_key = self._compute_input_hash(X, y, use_entropy_as_king)
        if cache_key and cache_key in self._CACHE:
            tprint_success(f"⚡ [DePrado] Cache Hit! Returning pre-computed selection for {len(X.columns)} features")
            return self._CACHE[cache_key]
        
        start_time = time.time()
        tprint_info("🚀 Starting De Prado Feature Selection Engine...")
        tprint_info(f"📊 Input: {len(X.columns)} features, {len(X)} samples")
        
        # Data quality checks
        if X.empty or len(X) == 0:
            tprint_error("❌ Empty feature matrix provided to De Prado engine")
            raise ValueError("Empty feature matrix")
        
        # ... (rest of method) ...

        
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            tprint_warning(f"⚠️ Found {missing_values} missing values, will fill with 0")
            X = X.fillna(0)
        
        # Check target variable
        unique_classes = len(np.unique(y))
        tprint_info(f"🎯 Target variable: {unique_classes} unique classes")
        
        if unique_classes < 2:
            tprint_error("❌ Target variable has less than 2 classes")
            raise ValueError("Target variable must have at least 2 classes")
        
        # 1. Cluster Features (Redundancy Control)
        tprint_info("🔍 Step 1: Finding optimal feature clusters (ONC)...")
        clustering_start = time.time()
        
        cluster_map = self._get_onc_clusters(X)
        self.cluster_labels_ = cluster_map
        
        clustering_time = time.time() - clustering_start
        tprint_info(f"⏱️  Clustering completed in {clustering_time:.2f}s")
        tprint_info(f"📊 Found {self.optimal_n_clusters_} clusters")
        
        # Report cluster sizes
        cluster_sizes = cluster_map.value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            tprint_info(f"   Cluster {cluster_id}: {size} features")
        
        # 2. Train ExtraTrees (The Analyst)
        tprint_info("🌳 Step 2: Training ExtraTrees for MDI analysis...")
        training_start = time.time()
        
        try:
            model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                bootstrap=False  # Use all samples for stability
            )
            
            # Handle class imbalance
            if len(np.unique(y)) == 2:
                class_weight = "balanced"
                tprint_info("⚖️  Using balanced class weights")
            else:
                class_weight = None
            
            model.set_params(class_weight=class_weight)
            
            tprint_info(f"🔄 Training {self.n_estimators} trees...")
            model.fit(X, y)
            
            training_time = time.time() - training_start
            tprint_success(f"✅ ExtraTrees training completed in {training_time:.2f}s")
            
        except Exception as e:
            tprint_error(f"❌ ExtraTrees training failed: {e}")
            raise
        
        # 3. Collect Advanced Metrics
        tprint_info("📈 Step 3: Computing advanced metrics (MDI, IC, Entropy)...")
        metrics_start = time.time()
        
        feature_names = X.columns.tolist()
        
        # 3a. MDI metrics
        mdi_metrics = self._compute_advanced_mdi(model, feature_names)
        gain = pd.Series(mdi_metrics["gain"])
        cover = pd.Series(mdi_metrics["cover"])
        
        # 3b. Hierarchy (Root Proximity)
        depth = self._get_tree_hierarchy(model, feature_names)
        
        # 3c. Information Coefficient (Spearman)
        tprint_info("📊 Computing Information Coefficient (Spearman IC)...")
        ic_scores = {}
        for col in X.columns:
            try:
                ic_val, _ = spearmanr(X[col], y)
                ic_scores[col] = abs(ic_val) if not np.isnan(ic_val) else 0.0
            except Exception:
                ic_scores[col] = 0.0
        ic_series = pd.Series(ic_scores)
        
        # 3d. Shannon Entropy
        tprint_info("📊 Computing Shannon Entropy (10-quantile discretization)...")
        entropy_scores = {}
        for col in X.columns:
            try:
                # Discretize into 10 quantiles
                discretized = pd.qcut(X[col], q=10, duplicates='drop')
                if discretized.nunique() < 2:
                    entropy_scores[col] = 0.0
                else:
                    value_counts = discretized.value_counts(normalize=True)
                    entropy_scores[col] = entropy(value_counts)
            except Exception:
                entropy_scores[col] = 0.0
        ent_series = pd.Series(entropy_scores)
        
        metrics_time = time.time() - metrics_start
        tprint_info(f"⏱️  Metrics computation completed in {metrics_time:.2f}s")
        
        # 4. Normalize and Score
        tprint_info("⚖️  Step 4: Computing composite scores...")
        scoring_start = time.time()
        
        def normalize_series(s, invert=False):
            s_range = s.max() - s.min()
            if s_range > EPS:
                if invert:
                    return (s.max() - s) / s_range
                return (s - s.min()) / s_range
            return pd.Series(1.0, index=s.index)

        score_gain = normalize_series(gain)
        score_depth = normalize_series(depth, invert=True)
        score_ic = normalize_series(ic_series)
        score_ent = normalize_series(ent_series)
        
        # Composite Score: weighted combination
        composite = (
            self.gain_weight * score_gain + 
            self.depth_weight * score_depth + 
            self.ic_weight * score_ic + 
            self.entropy_weight * score_ent
        )
        
        scoring_time = time.time() - scoring_start
        tprint_info(f"⏱️  Scoring completed: Gain={self.gain_weight:.1f}, Depth={self.depth_weight:.1f}, IC={self.ic_weight:.1f}, Ent={self.entropy_weight:.1f}")
        
        # 5. Store Results
        self.feature_stats_ = pd.DataFrame({
            "Cluster": cluster_map,
            "Gain": gain,
            "IC": ic_series,
            "Entropy": ent_series,
            "MeanDepth": depth,
            "GainScore": score_gain,
            "DepthScore": score_depth,
            "ICScore": score_ic,
            "EntropyScore": score_ent,
            "CompositeScore": composite
        })
        
        # 6. Intra-Cluster Selection: Pick the "King" of each cluster
        tprint_info(f"👑 Step 6: Picking kings (use_entropy_as_king={use_entropy_as_king})...")
        selection_start = time.time()
        
        selected_features = []
        cluster_summary = {}
        
        for cluster_id in sorted(cluster_map.unique()):
            cluster_features = self.feature_stats_[self.feature_stats_["Cluster"] == cluster_id]
            if len(cluster_features) == 0: continue
            
            # Select best feature from cluster
            if use_entropy_as_king:
                # Sole criterion is entropy
                best_feature = cluster_features.loc[cluster_features["EntropyScore"].idxmax()]
            else:
                # Use composite score
                best_feature = cluster_features.loc[cluster_features["CompositeScore"].idxmax()]
            
            selected_features.append(best_feature.name)
            
            cluster_summary[cluster_id] = {
                "n_features": len(cluster_features),
                "best_feature": best_feature.name,
                "best_score": best_feature["CompositeScore"] if not use_entropy_as_king else best_feature["EntropyScore"]
            }
            
            tprint_info(f"   Cluster {cluster_id}: {len(cluster_features)} → {best_feature.name}")
        
        self.selected_features_ = selected_features
        
        # Enhanced detailed reporting
        if True:
            self._print_detailed_deprado_report(X)
        
        selection_time = time.time() - selection_start
        total_time = time.time() - start_time
        
        tprint_success(f"✅ De Prado Selection completed in {total_time:.2f}s")
        tprint_success(f"📊 Selected {len(selected_features)}/{len(X.columns)} features")
        
        # --- UPDATE CACHE ---
        if cache_key:
            self._CACHE[cache_key] = selected_features
            tprint_info(f"⚡ [DePrado] Caching results for future use")
        
        return selected_features
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Get comprehensive feature statistics.
        
        Returns:
            DataFrame with feature statistics for all features
        """
        if self.feature_stats_ is None:
            raise ValueError("Feature selection not run. Call run_selection() first.")
        return self.feature_stats_.copy()
    
    def get_selected_features(self) -> List[str]:
        """
        Get list of selected feature names.
        
        Returns:
            List of selected feature names
        """
        if self.selected_features_ is None:
            raise ValueError("Feature selection not run. Call run_selection() first.")
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
        """
        Get detailed report for selected features.
        
        Returns:
            DataFrame with detailed stats for selected features only
        """
        if self.feature_stats_ is None:
            raise ValueError("Feature selection not run. Call run_selection() first.")
        
        selected_stats = self.feature_stats_.loc[self.selected_features_].copy()
        return selected_stats.sort_values('CompositeScore', ascending=False)

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
                tprint_info(f"   👑 KING: {king_feature.name} (score: {king_feature['CompositeScore']:.3f}) ✅ SELECTED")
                tprint_info(f"      📊 Gain: {king_feature['Gain']:.4f}, Cover: {king_feature['Cover']:.4f}, Depth: {king_feature['MeanDepth']:.2f}")
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
    random_state: int = 42
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
        
    Returns:
        Tuple of (selected_features_df, fitted_engine)
    """
    engine = DePradoFeatureEngine(
        n_estimators=n_estimators,
        max_clusters=max_clusters,
        gain_weight=gain_weight,
        depth_weight=depth_weight,
        random_state=random_state
    )
    
    selected_features = engine.run_selection(X, y)
    X_selected = X[selected_features].copy()
    
    return X_selected, engine

