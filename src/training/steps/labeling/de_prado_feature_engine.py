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
from sklearn.metrics import silhouette_score
from scipy.stats import norm
import warnings
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

EPS = 1e-12

class DePradoFeatureEngine:
    """
    Complete Feature Selection Engine integrating:
    1. ONC Clustering (Redundancy Filter)
    2. Advanced MDI (Gain/Cover - Power Filter)  
    3. Root Proximity (Hierarchy Filter)
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_clusters: int = 12,
        min_cluster_size: int = 2,
        random_state: int = 42,
        gain_weight: float = 0.5,
        depth_weight: float = 0.5,
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
            gain_weight: Weight for gain in composite score (default: 0.5)
            depth_weight: Weight for depth proximity in composite score (default: 0.5)
            min_samples_leaf: Minimum samples per leaf in ExtraTrees
            max_features: Max features considered for each split
        """
        self.n_estimators = n_estimators
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.gain_weight = gain_weight
        self.depth_weight = depth_weight
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
        Finds optimal feature clusters using Silhouette Score.
        
        Args:
            X: Feature matrix
            
        Returns:
            Series of cluster labels indexed by feature names
        """
        tprint_info("🔍 Finding optimal feature clusters (ONC)...")
        
        # Compute correlation matrix
        corr = X.corr().fillna(0)
        
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
        
        # Find optimal number of clusters
        best_k, best_score = 2, -1
        scores_history = {}
        
        # Test cluster sizes from 2 to max_clusters or n_features//2
        max_k = min(self.max_clusters, max(2, len(X.columns) // 2))
        
        for k in range(2, max_k + 1):
            try:
                clusterer = FeatureAgglomeration(n_clusters=k, linkage='average')
                clusterer.fit(X)
                cluster_labels = clusterer.labels_
                
                # Debug: Check if clustering actually produced k clusters
                unique_labels = np.unique(cluster_labels)
                if len(unique_labels) != k:
                    tprint_warning(f"Clustering for k={k}: Expected {k} clusters, got {len(unique_labels)}")
                
                # Compute silhouette score
                if len(unique_labels) > 1:
                    # For precomputed distance, we need to pass the distance matrix
                    # But cluster_labels corresponds to samples, not features
                    # We should compute silhouette on the feature correlation distance
                    score = silhouette_score(dist.T, cluster_labels, metric='precomputed')
                    scores_history[k] = score
                    tprint_info(f"   k={k}: silhouette={score:.3f}")
                    
                    if score > best_score:
                        best_score, best_k = score, k
                else:
                    tprint_warning(f"Clustering for k={k}: Only 1 cluster found")
                        
            except Exception as e:
                tprint_warning(f"Clustering failed for k={k}: {e}")
                # Debug: Print more details about the data
                if k == 2:  # Only print for first failure to avoid spam
                    tprint_warning(f"   Data shape: {X.shape}")
                    tprint_warning(f"   Distance matrix shape: {dist.shape}")
                    tprint_warning(f"   Distance matrix stats: min={dist.values.min():.3f}, max={dist.values.max():.3f}, mean={dist.values.mean():.3f}")
                continue
        
        if best_k == 2 and best_score < 0:
            if len(X.columns) > 20:
                tprint_warning(f"ONC Clustering quality low (best silhouette {best_score:.3f}). Forcing 5 clusters for diversity.")
                best_k = 5
            else:
                tprint_warning("Clustering failed (silhouette < 0). Using single cluster.")
                best_k = 1
        
        # Final clustering with optimal k
        if best_k == 1:
            final_labels = np.zeros(len(X.columns))
        else:
            final_clusterer = FeatureAgglomeration(n_clusters=best_k, linkage='average')
            final_clusterer.fit(X)
            final_labels = final_clusterer.labels_
        
        self.optimal_n_clusters_ = best_k
        self.silhouette_scores_ = scores_history
        
        tprint_success(f"✅ ONC: {best_k} optimal clusters found (silhouette: {best_score:.3f})")
        
        return pd.Series(final_labels, index=X.columns)
    
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
    
    def run_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Run complete De Prado feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        start_time = time.time()
        tprint_info("🚀 Starting De Prado Feature Selection Engine...")
        tprint_info(f"📊 Input: {len(X.columns)} features, {len(X)} samples")
        
        # Data quality checks
        if X.empty or len(X) == 0:
            tprint_error("❌ Empty feature matrix provided to De Prado engine")
            raise ValueError("Empty feature matrix")
        
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
        """
        Run complete De Prado feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            List of selected feature names
        """
        tprint_info("🚀 Starting De Prado Feature Selection Engine...")
        tprint_info(f"📊 Input: {len(X.columns)} features, {len(X)} samples")
        
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
            
            # Check for empty data
            if X.empty or len(X) == 0:
                tprint_error("❌ Empty feature matrix provided to De Prado engine")
                raise ValueError("Empty feature matrix")
            
            tprint_info(f"🔄 Training {self.n_estimators} trees...")
            model.fit(X, y)
            
            training_time = time.time() - training_start
            tprint_success(f"✅ ExtraTrees training completed in {training_time:.2f}s")
            
        except Exception as e:
            tprint_error(f"❌ ExtraTrees training failed: {e}")
            raise
            
        except Exception as e:
            tprint_error(f"❌ ExtraTrees training failed: {e}")
            raise
        
        # 3. Collect Advanced Metrics
        tprint_info("📈 Step 3: Computing advanced MDI metrics...")
        metrics_start = time.time()
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Compute MDI metrics
        mdi_metrics = self._compute_advanced_mdi(model, feature_names)
        gain = pd.Series(mdi_metrics["gain"])
        cover = pd.Series(mdi_metrics["cover"])
        
        # Compute hierarchy (Root Proximity)
        tprint_info("🌳 Computing tree hierarchy metrics...")
        depth = self._get_tree_hierarchy(model, feature_names)
        
        metrics_time = time.time() - metrics_start
        tprint_info(f"⏱️  Metrics computation completed in {metrics_time:.2f}s")
        tprint_info(f"📊 Gain range: {gain.min():.6f} - {gain.max():.6f}")
        tprint_info(f"📊 Depth range: {depth.min():.1f} - {depth.max():.1f}")
        
        # 4. Normalize and Score
        tprint_info("⚖️  Step 4: Computing composite scores...")
        scoring_start = time.time()
        
        # Normalize Gain (higher is better)
        gain_range = gain.max() - gain.min()
        if gain_range > EPS:
            score_gain = (gain - gain.min()) / gain_range
        else:
            tprint_warning("⚠️  All gain values are equal, using uniform scores")
            score_gain = pd.Series(1.0, index=gain.index)
        
        # Normalize Depth (lower depth = higher proximity to root = better)
        depth_range = depth.max() - depth.min()
        if depth_range > EPS:
            score_depth = (depth.max() - depth) / depth_range
        else:
            tprint_warning("⚠️  All depth values are equal, using uniform scores")
            score_depth = pd.Series(1.0, index=depth.index)
        
        # Composite Score: weighted combination
        composite = (self.gain_weight * score_gain) + (self.depth_weight * score_depth)
        
        scoring_time = time.time() - scoring_start
        tprint_info(f"⏱️  Scoring completed in {scoring_time:.2f}s")
        tprint_info(f"⚖️  Weights: Gain={self.gain_weight:.1f}, Depth={self.depth_weight:.1f}")
        tprint_info(f"📊 Composite score range: {composite.min():.3f} - {composite.max():.3f}")
        
        # 5. Store Results
        tprint_info("💾 Step 5: Storing feature statistics...")
        
        self.feature_stats_ = pd.DataFrame({
            "Cluster": cluster_map,
            "Gain": gain,
            "Cover": cover,
            "MeanDepth": depth,
            "GainScore": score_gain,
            "DepthScore": score_depth,
            "CompositeScore": composite
        })
        
        # 6. Intra-Cluster Selection: Pick the "King" of each cluster
        tprint_info("👑 Step 6: Selecting king features from each cluster...")
        selection_start = time.time()
        
        selected_features = []
        cluster_summary = {}
        total_clusters_processed = 0
        skipped_clusters = 0
        cluster_summary = {}
        
        for cluster_id in sorted(cluster_map.unique()):
            cluster_features = self.feature_stats_[self.feature_stats_["Cluster"] == cluster_id]
            
            if len(cluster_features) == 0:
                continue
            
            total_clusters_processed += 1
            
            # Skip clusters that are too small (optional)
            if len(cluster_features) < self.min_cluster_size and len(cluster_features) < len(X.columns) * 0.05:
                tprint_warning(f"   Skipping small cluster {cluster_id} ({len(cluster_features)} features)")
                skipped_clusters += 1
                continue
            
            # Select best feature from cluster
            best_feature = cluster_features.loc[cluster_features["CompositeScore"].idxmax()]
            selected_features.append(best_feature.name)
            
            cluster_summary[cluster_id] = {
                "n_features": len(cluster_features),
                "best_feature": best_feature.name,
                "best_score": best_feature["CompositeScore"],
                "avg_gain": cluster_features["Gain"].mean(),
                "avg_depth": cluster_features["MeanDepth"].mean()
            }
            
            tprint_info(f"   Cluster {cluster_id}: {len(cluster_features)} → {best_feature.name} (score: {best_feature['CompositeScore']:.3f})")
            
            cluster_summary[cluster_id] = {
                'n_features': len(cluster_features),
                'best_feature': best_feature.name,
                'best_score': best_feature['CompositeScore'],
                'avg_gain': cluster_features['Gain'].mean(),
                'avg_depth': cluster_features['MeanDepth'].mean()
            }
        
        self.selected_features_ = selected_features
        
        # Enhanced detailed reporting
        if True:
            self._print_detailed_deprado_report(X)
        
        selection_time = time.time() - selection_start
        total_time = time.time() - start_time
        
        # 7. Report Results
        tprint_success(f"✅ De Prado Selection completed in {total_time:.2f}s")
        tprint_success(f"📊 Selected {len(selected_features)}/{len(X.columns)} features ({len(selected_features)/len(X.columns):.1%})")
        tprint_info(f"👑 Clusters processed: {total_clusters_processed}, Skipped: {skipped_clusters}")
        tprint_info(f"⏱️  Feature selection: {selection_time:.2f}s")
        tprint_info(f"📊 Average cluster size: {np.mean([s['n_features'] for s in cluster_summary.values()]):.1f}")
        
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


