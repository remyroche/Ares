"""
Causal Feature Sieve - 4-Sieve Feature Selection Pipeline (2026 Production Standard)

Implements refined feature selection addressing:
- Effective Sample Size (T_eff) 
- Horizon scaling
- Dominance-Weighted Stability

Geometry-specific configurations for 12-bar (impulse) vs 48-bar (structural) horizons.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.linear_model import ElasticNetCV, ElasticNet, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import lightgbm as lgb

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

@dataclass
class GeometryConfig:
    """Geometry-specific configuration for CausalFeatureSieve"""
    horizon_bars: int
    max_clusters: int
    cv_folds: int
    l1_ratio: float
    alpha_rule: str  # 'min' or '1se'
    instability_threshold: float
    dist_metric: str = 'angular'
    
    def __post_init__(self):
        # Calculate T_eff for this geometry
        self.T_eff = self.T_eff if hasattr(self, 'T_eff') else None

class CausalFeatureSieve:
    """
    Refined 4-Sieve Feature Selection Pipeline (2026 Production Standard).
    Addresses Effective Sample Size, Horizon scaling, and Dominance-Weighted Stability.
    """
    
    # Geometry-specific configurations
    GEOMETRY_CONFIGS = {
        '12_bar': GeometryConfig(
            horizon_bars=12,
            max_clusters=15,
            cv_folds=5,
            l1_ratio=1.0,  # Pure Lasso
            alpha_rule='min',
            instability_threshold=0.50,
            dist_metric='angular'
        ),
        '48_bar': GeometryConfig(
            horizon_bars=48,
            max_clusters=20,
            cv_folds=6,
            l1_ratio=0.8,  # ElasticNet mix
            alpha_rule='1se',
            instability_threshold=0.40,
            dist_metric='angular'
        )
    }
    
    def __init__(self, geometry: str = '12_bar', seed: int = 42):
        """
        Initialize CausalFeatureSieve for specific geometry.
        
        Args:
            geometry: '12_bar' or '48_bar'
            seed: Random seed for reproducibility
        """
        if geometry not in self.GEOMETRY_CONFIGS:
            raise ValueError(f"Geometry must be '12_bar' or '48_bar', got {geometry}")
            
        self.geometry = geometry
        self.config = self.GEOMETRY_CONFIGS[geometry]
        self.seed = seed
        self.cv_folds = self.config.cv_folds
        
        # Diagnostic logging
        self.logger = logging.getLogger(__name__)
        np.random.seed(self.seed)
        
        tprint_info(f"🔧 CausalFeatureSieve initialized for {geometry} geometry")
        tprint_info(f"   - Horizon: {self.config.horizon_bars} bars")
        tprint_info(f"   - Max clusters: {self.config.max_clusters}")
        tprint_info(f"   - CV folds: {self.config.cv_folds}")
        tprint_info(f"   - L1 ratio: {self.config.l1_ratio}")
        tprint_info(f"   - Instability threshold: {self.config.instability_threshold}")

    def sieve_1_onc(self, X: pd.DataFrame, T: int) -> pd.DataFrame:
        """
        Sieve 1: Optimal Number of Clusters (ONC) via MP-Adjusted Hierarchical Linkage.
        """
        regime_features = self._detect_regime_features(X)
        tprint_info(f"🔍 Sieve 1: ONC Clustering ({len(X.columns)} features)")
        if regime_features:
            tprint_info(f"   🎭 Including {len(regime_features)} regime features in clustering")
        
        corr = X.corr().fillna(0.0)
        
        # Angular distance: 0.5 * (1 - ρ)
        if self.config.dist_metric == 'angular':
            dist = np.sqrt(0.5 * (1 - corr))
        else:
            dist = np.sqrt(1 - corr**2)
        
        # Issue 1 Fix: Effective Sample Size (T_eff)
        T_eff = T / self.config.horizon_bars
        mp_upper_bound = (1 + np.sqrt(len(X.columns) / T_eff))**2
        
        eigenvals = np.linalg.eigvalsh(corr.values)
        significant_factors = np.sum(eigenvals > mp_upper_bound)
        
        tprint_info(f"   📊 T_eff: {T_eff:.2f}, MP bound: {mp_upper_bound:.4f}")
        tprint_info(f"   🎯 Significant factors: {significant_factors}")
        
        # Issue 2 Fix: Robust Guardrail for Search Range
        low = max(2, significant_factors - 2)
        high = min(self.config.max_clusters, int(np.sqrt(len(X.columns))), len(X.columns))
        if low >= high:
            low, high = 2, min(10, len(X.columns))
        search_range = range(low, high + 1)
        
        tprint_info(f"   🔍 Cluster search range: {low} to {high}")
        
        # Hierarchical Linkage
        condensed = squareform(dist.values, checks=False)
        Z = linkage(condensed, method='average')

        best_k, best_score = 2, -1
        for k in search_range:
            labels = fcluster(Z, k, criterion='maxclust')
            if len(np.unique(labels)) < 2:  # Need at least 2 clusters
                continue
            score = silhouette_score(dist, labels, metric='precomputed')
            if score > best_score:
                best_k, best_score = k, score

        # Issue 3 Fix: Horizon-Aware Adjustment for K
        adj_factor = np.sqrt(self.config.horizon_bars / 12)
        best_k = max(2, int(best_k / adj_factor))
        
        tprint_info(f"   ✅ Final optimized K: {best_k} (Silhouette: {best_score:.4f})")
        
        final_labels = fcluster(Z, best_k, criterion='maxclust')
        medoids = []
        for cluster_id in np.unique(final_labels):
            idx = np.where(final_labels == cluster_id)[0]
            sub_dist = dist.iloc[idx, idx]
            medoid_idx = sub_dist.sum(axis=1).idxmin()
            # medoid_idx is already the column name/index we want
            medoids.append(medoid_idx)

        selected_features = medoids
        tprint_info(f"   📉 ONC reduced: {len(X.columns)} → {len(selected_features)} features")
        
        return X[selected_features]

    def sieve_2_elastic_1se(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Sieve 2: Parsimony via ElasticNet (1-SE Rule) -> LASSO Pruning.
        Includes a subsequent LASSO step to strictly prune features not kept.
        """
        regime_features = self._detect_regime_features(X)
        tprint_info(f"🔍 Sieve 2: ElasticNet + LASSO Selection ({len(X.columns)} features)")
        if regime_features:
            tprint_info(f"   🎭 Processing {len(regime_features)} regime features")
        
        # Create purged time series CV
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_folds,
            gap=1,  # Small gap to prevent leakage
            test_size=None
        )
        
        # ElasticNet improves stability over pure LASSO in correlated clusters
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNetCV(
                l1_ratio=[self.config.l1_ratio], 
                cv=tscv, 
                n_alphas=100, 
                random_state=self.seed,
                max_iter=2000
            ))
        ])

        pipe.fit(X, y)
        ecv = pipe.named_steps['en']

        # Robust 1-SE alpha selection
        # Handle different shapes of mse_path_
        if ecv.mse_path_.ndim == 3:
            mse_mean = ecv.mse_path_.mean(axis=2).mean(axis=1)  # Average across folds and l1_ratios
            mse_std = ecv.mse_path_.std(axis=2).mean(axis=1) / np.sqrt(self.config.cv_folds)
        else:
            mse_mean = ecv.mse_path_.mean(axis=1)  # Average across folds only
            mse_std = ecv.mse_path_.std(axis=1) / np.sqrt(self.config.cv_folds)
        
        idx_min = mse_mean.argmin()
        
        if self.config.alpha_rule == '1se':
            threshold = mse_mean[idx_min] + mse_std[idx_min]
            eligible = np.where(mse_mean <= threshold)[0]
            alpha_1se = ecv.alphas_[eligible[-1]] if len(eligible) > 0 else ecv.alphas_[idx_min]
        else:  # 'min'
            alpha_1se = ecv.alphas_[idx_min]

        final_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=alpha_1se, l1_ratio=ecv.l1_ratio_, random_state=self.seed, max_iter=2000))
        ])
        final_pipe.fit(X, y)
        
        coefs = final_pipe.named_steps['en'].coef_
        en_selected = X.columns[coefs != 0].tolist()

        tprint_info(f"   🎯 ElasticNet selected: {len(en_selected)} features (alpha: {alpha_1se:.6f})")

        if not en_selected:
            tprint_warning("   ⚠️ ElasticNet pruned all features. Returning best single feature.")
            # Fallback to single best correlation
            corrs = X.corrwith(y).abs()
            return [corrs.idxmax()]

        # --- Sub-step: LASSO Pruning ---
        # Run strict LASSO on the EN-selected features to prune further
        if len(en_selected) > 1:
            X_en = X[en_selected]
            lasso_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', LassoCV(cv=tscv, random_state=self.seed, max_iter=2000))
            ])
            lasso_pipe.fit(X_en, y)

            lasso_coefs = lasso_pipe.named_steps['lasso'].coef_
            lasso_selected = X_en.columns[lasso_coefs != 0].tolist()

            if lasso_selected:
                tprint_info(f"   ✂️ LASSO Pruning reduced: {len(en_selected)} → {len(lasso_selected)} features")
                selected_features = lasso_selected
            else:
                tprint_warning("   ⚠️ LASSO pruned all. Reverting to ElasticNet selection.")
                selected_features = en_selected
        else:
            selected_features = en_selected
        
        tprint_info(f"   📉 Sieve 2 Final: {len(X.columns)} → {len(selected_features)} features")
        
        return selected_features

    def _get_tree_hierarchy_analysis(self, models: List, feature_names: List[str]) -> pd.Series:
        """
        Calculate mean first split depth (root proximity) from De Prado.
        Features used earlier in trees (shallower depth) are more structurally important.
        Optimized for LightGBM tree structure.
        """
        depths = {name: [] for name in feature_names}
        max_depth_overall = 0
        
        for model in models:
            try:
                # Extract tree structure from LightGBM
                tree_dump = model._Booster._dump_model()['tree_info']
                
                for tree_info in tree_dump:
                    tree = tree_info['tree_structure']
                    max_depth_overall = max(max_depth_overall, self._get_tree_depth(tree))
                    first_occurrence = {}
                    
                    def walk_node(node: dict, current_depth: int):
                        """Walk tree recursively to find first feature usage."""
                        if 'split_feature' in node:
                            feature_idx = node['split_feature']
                            if feature_idx not in first_occurrence:
                                first_occurrence[feature_idx] = current_depth
                            # Continue walking
                            if 'left_child' in node:
                                walk_node(node['left_child'], current_depth + 1)
                            if 'right_child' in node:
                                walk_node(node['right_child'], current_depth + 1)
                    
                    walk_node(tree, 0)
                    
                    # Record depths for this tree
                    for idx, depth in first_occurrence.items():
                        if idx < len(feature_names):
                            depths[feature_names[idx]].append(depth)
                        
            except Exception as e:
                # Fallback: assign max depth for all features if tree parsing fails
                for name in feature_names:
                    depths[name].append(max_depth_overall)
        
        # Calculate mean depths
        mean_depths = {}
        for name, depth_list in depths.items():
            if depth_list:
                mean_depths[name] = np.median(depth_list)
            else:
                mean_depths[name] = max_depth_overall
        
        return pd.Series(mean_depths)
    
    def _get_tree_depth(self, tree: dict) -> int:
        """Calculate maximum depth of a tree."""
        def get_depth(node: dict) -> int:
            if 'split_feature' not in node:
                return 0
            left_depth = get_depth(node.get('left_child', {}))
            right_depth = get_depth(node.get('right_child', {}))
            return 1 + max(left_depth, right_depth)
        
        return get_depth(tree)
    
    def _generate_enhanced_lgbm_importance(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Enhanced LGBM importance with OOF predictions, IC, stability metrics, and tree hierarchy.
        Optimized for memory and computational efficiency.
        """
        tprint_info(f"   📊 Generating Enhanced LGBM importance ({len(X.columns)} features)")
        
        # Determine task type
        is_classifier = len(y.unique()) <= 2
        
        # Enhanced LightGBM params (from De Prado) - optimized for speed
        params = {
            'n_estimators': 500,      # Reduced from 1000 for speed
            'learning_rate': 0.05,
            'max_depth': 4,           # Controlled depth for hierarchy
            'num_leaves': 15,         # Reduced for memory efficiency
            'reg_alpha': 5.0,         # Stronger L1 (from De Prado)
            'reg_lambda': 10.0,       # Stronger L2 (from De Prado)
            'min_child_samples': 20,
            'min_split_gain': 1e-3,
            'colsample_bytree': 0.7,
            'subsample': 0.8,
            'importance_type': 'gain',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': self.seed
        }
        
        try:
            # Use TimeSeriesSplit for temporal safety
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5, gap=1)  # Add gap to prevent leakage
            
            # Pre-allocate arrays for memory efficiency
            n_features = len(X.columns)
            fold_importances = np.zeros((5, n_features))
            fold_ics = np.zeros((5, n_features))
            oof_preds = np.full(len(X), np.nan)
            
            # Store models for tree hierarchy analysis
            trained_models = []
            
            # Convert to numpy for speed
            X_values = X.values
            y_values = y.values
            feature_names = X.columns.tolist()
            
            if sample_weight is not None:
                w_values = sample_weight.values
            else:
                w_values = None
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_values)):
                # Slice arrays (faster than DataFrame.iloc)
                X_train, X_val = X_values[train_idx], X_values[val_idx]
                y_train, y_val = y_values[train_idx], y_values[val_idx]
                
                if w_values is not None:
                    w_train = w_values[train_idx]
                else:
                    w_train = None
                
                # Train model
                if is_classifier:
                    model = lgb.LGBMClassifier(**params)
                else:
                    model = lgb.LGBMRegressor(**params)
                
                model.fit(X_train, y_train, sample_weight=w_train)
                trained_models.append(model)  # Store for hierarchy analysis
                
                # Store importance
                fold_importances[fold] = model.feature_importances_
                
                # OOF predictions (vectorized)
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X_val)[:, 1]
                else:
                    preds = model.predict(X_val)
                oof_preds[val_idx] = preds
                
                # Vectorized feature-level IC calculation
                if len(X_val) > 1:  # Need at least 2 points for correlation
                    for i, col in enumerate(feature_names):
                        try:
                            ic_val, _ = spearmanr(X_val[:, i], y_val)
                            fold_ics[fold, i] = ic_val if not np.isnan(ic_val) else 0.0
                        except:
                            fold_ics[fold, i] = 0.0
                else:
                    fold_ics[fold, :] = 0.0
            
            # Aggregate metrics (vectorized)
            mean_importance = pd.Series(fold_importances.mean(axis=0), index=feature_names)
            median_feature_ic = pd.Series(np.median(fold_ics, axis=0), index=feature_names)
            
            # OOF IC (model-level)
            valid_mask = ~np.isnan(oof_preds) & ~np.isnan(y_values)
            oof_ic = 0.0
            if valid_mask.sum() > 10:
                oof_ic, _ = spearmanr(y_values[valid_mask], oof_preds[valid_mask])
            
            # Calculate stability (coefficient of variation)
            importance_std = fold_importances.std(axis=0)
            importance_mean = fold_importances.mean(axis=0)
            stability = importance_mean / (importance_std + 1e-9)
            stability_series = pd.Series(stability, index=feature_names)
            
            # Top-K frequency (how often features appear in top 20%)
            top_k_threshold = np.percentile(fold_importances, 80, axis=1, keepdims=True)
            top_k_mask = fold_importances >= top_k_threshold
            topk_freq = pd.Series(top_k_mask.mean(axis=0), index=feature_names)
            
            # Tree hierarchy analysis (root proximity)
            depth_scores = self._get_tree_hierarchy_analysis(trained_models, feature_names)
            
            tprint_info(f"   ✅ Enhanced LGBM importance: OOF IC={oof_ic:.4f}, Tree depth analysis computed")
            
            return {
                'mean_importance': mean_importance,
                'median_feature_ic': median_feature_ic,
                'oof_ic': oof_ic,
                'stability': stability_series,
                'topk_freq': topk_freq,
                'depth_scores': depth_scores,  # NEW: Root proximity analysis
                'oof_predictions': pd.Series(oof_preds, index=X.index),
                'trained_models': trained_models  # Store models for debugging
            }
            
        except Exception as e:
            tprint_error(f"   ❌ Enhanced LGBM importance failed: {e}")
            # Fallback to simple correlation
            importance_scores = {}
            for col in X.columns:
                try:
                    corr = X[col].corr(y)
                    importance_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    importance_scores[col] = 0.0
            
            fallback_importance = pd.Series(importance_scores)
            n_features = len(X.columns)
            return {
                'mean_importance': fallback_importance,
                'median_feature_ic': fallback_importance * 0.5,  # Estimate
                'oof_ic': 0.0,
                'stability': pd.Series(1.0, index=X.columns),
                'topk_freq': pd.Series(0.5, index=X.columns),
                'depth_scores': pd.Series(np.full(n_features, 2.0), index=X.columns),  # Default depth
                'oof_predictions': pd.Series(0.5, index=X.index),
                'trained_models': []
            }

    def sieve_3_4_dominance_stability(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> List[str]:
        """
        Sieve 3 & 4: Dominance-Weighted Rank Stability.
        Issue 5 Fix: Penalize low importance even if stable.
        """
        tprint_info(f"🔍 Sieve 3/4: LGBM Importance + Stability Analysis ({len(X.columns)} features)")
        
        # Generate MDA importance DataFrame
        mda_importance_df = self._generate_mda_importance(X, y, sample_weight)
        
        if mda_importance_df.empty:
            tprint_warning("   ⚠️ No importance available, returning all features")
            return X.columns.tolist()
        
        # Create multiple runs for stability analysis (Varying seeds)
        n_runs = 5
        mda_runs = []
        
        for run in range(n_runs):
            # Update seed in self temporarily or pass to func
            original_seed = self.seed
            self.seed = original_seed + run
            run_importance = self._generate_mda_importance(X, y, sample_weight)
            self.seed = original_seed # Restore

            if not run_importance.empty:
                mda_runs.append(run_importance)
        
        if len(mda_runs) < 2:
            tprint_warning("   ⚠️ Insufficient runs for stability analysis")
            mean_importance = mda_importance_df.iloc[:, 0]
            instability = pd.Series(0.0, index=mda_importance_df.index)
        else:
            # Combine multiple runs
            combined_importance = pd.concat(mda_runs, axis=1)
            mean_importance = combined_importance.mean(axis=1)
            
            # Calculate rank stability across runs
            ranks = combined_importance.rank(axis=0, ascending=False, method='min')
            # Normalized Std of Ranks
            std_rank = ranks.std(axis=1)
            mean_rank = ranks.mean(axis=1)
            
            # Instability Index: Std / Mean (Coefficient of Variation of Rank)
            instability = std_rank / (mean_rank + 1e-9)
        
        # Issue 5 Fix: Dominance weighting
        # Score = Importance / (Instability + eps)
        dominance_stability_score = mean_importance / (instability + 1e-9)
        
        results = pd.DataFrame({
            'mean_importance': mean_importance,
            'instability_index': instability,
            'dom_stab_score': dominance_stability_score,
            'is_stable': instability <= self.config.instability_threshold
        }).sort_values('dom_stab_score', ascending=False)
        
        # Select top features that are stable
        # Filter by stability threshold AND take top N (e.g. top 50%)
        stable_candidates = results[results['is_stable']]

        # Further pruning: Keep top 50% of stable features by score
        n_keep = max(1, int(len(stable_candidates) * 0.5))
        final_selected = stable_candidates.head(n_keep)

        stable_features = final_selected.index.tolist()
        
        tprint_info(f"   📊 Runs: {n_runs}, Stable Threshold: {self.config.instability_threshold}")
        tprint_info(f"   📉 Stability reduced: {len(X.columns)} → {len(stable_features)} features")
        
        return stable_features


    def _detect_regime_features(self, X: pd.DataFrame) -> List[str]:
        """
        Detect regime-related features that may need special handling.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of regime feature names
        """
        regime_patterns = [
            'slope_short', 'adx_proxy', 'momentum_short', 'snr',
            'choppiness_index', 'variance_ratio', 'efficiency_ratio',
            'permutation_entropy', 'hour_sin', 'hour_cos', 'day_of_week',
            'is_weekend', 'time_since_last_vol_spike', 'time_since_last_large_candle',
            'momentum_agreement', 'momentum_agreement_abs', 'momentum_weighted_agreement',
            'trend_consistency_12', 'vol_long', 'vol_ratio', 'regime_sadf',
            'sadf_score_norm', 'cusum_score_norm', 'volatility_zscore',
            'volatility_regime', 'frac_vol', 'innov_vol'
        ]
        
        detected = []
        for col in X.columns:
            for pattern in regime_patterns:
                if pattern in col:
                    detected.append(col)
                    break
        
        return detected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply the complete 4-sieve pipeline to features.
        
        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Optional sample weights
            
        Returns:
            Selected feature matrix
        """
        tprint_info(f"🚀 CausalFeatureSieve: {self.geometry} pipeline start")
        tprint_info(f"📊 Input: {len(X.columns)} features, {len(X)} samples")
        
        # Detect regime features for special handling
        regime_features = self._detect_regime_features(X)
        if regime_features:
            tprint_info(f"🎭 Detected {len(regime_features)} regime features: {regime_features[:5]}...")
        
        initial_features = X.columns.tolist()
        T = len(X)
        
        # Sieve 1: ONC Clustering
        X_sieve1 = self.sieve_1_onc(X, T)
        if X_sieve1.empty:
            tprint_error("❌ Sieve 1 produced empty feature set")
            return X.iloc[:, :0]  # Return empty DataFrame
        
        # Sieve 2: ElasticNet + LASSO Selection
        selected_sieve2 = self.sieve_2_elastic_1se(X_sieve1, y)
        if not selected_sieve2:
            tprint_error("❌ Sieve 2 produced empty feature set")
            return X.iloc[:, :0]
        
        X_sieve2 = X_sieve1[selected_sieve2]
        
        # Sieve 3/4: LGBM Gain + Stability
        selected_sieve4 = self.sieve_3_4_dominance_stability(X_sieve2, y, sample_weight)
        if not selected_sieve4:
            tprint_error("❌ Sieve 3/4 produced empty feature set")
            return X.iloc[:, :0]
        
        X_final = X_sieve2[selected_sieve4]
        
        # Summary
        reduction_rate = 1 - len(X_final.columns) / len(initial_features)
        tprint_success(f"✅ CausalFeatureSieve: {self.geometry} complete!")
        tprint_success(f"📉 Feature reduction: {len(initial_features)} → {len(X_final.columns)} ({reduction_rate:.1%})")
        tprint_success(f"🎯 Final features: {X_final.columns.tolist()}")
        
        return X_final

    def _compute_shannon_entropy(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute Shannon entropy for each feature (from De Prado).
        Optimized with vectorized operations.
        """
        entropy_scores = {}
        
        # Vectorized entropy calculation
        for col in X.columns:
            try:
                # Use pandas qcut for efficient discretization
                disc = pd.qcut(X[col], q=10, duplicates='drop')
                if disc.nunique() < 2:
                    entropy_scores[col] = 0.0
                else:
                    # Vectorized probability calculation
                    probs = disc.value_counts(normalize=True).values
                    entropy_scores[col] = entropy(probs)
            except Exception:
                entropy_scores[col] = 0.0
        
        return pd.Series(entropy_scores)
    
    def _rank_normalize(self, s: pd.Series, invert: bool = False) -> pd.Series:
        """
        Rank normalization (from De Prado). Optimized implementation.
        """
        if s.empty:
            return s
        
        # Vectorized rank calculation
        ranks = rankdata(s, method='average')
        norm = (ranks - 1) / (len(s) - 1 + 1e-9)
        
        if invert:
            return pd.Series(1.0 - norm, index=s.index)
        return pd.Series(norm, index=s.index)
    
    def _compute_enhanced_composite_score(self, importance: pd.Series, ic: pd.Series, 
                                        entropy: pd.Series, stability: pd.Series,
                                        topk_freq: pd.Series, depth: pd.Series) -> pd.Series:
        """
        Enhanced composite scoring with De Prado metrics including root proximity.
        Geometry-specific weighting for optimal performance.
        """
        # Normalize all scores (vectorized)
        score_importance = self._rank_normalize(importance)
        score_ic = self._rank_normalize(ic.abs())  # Absolute IC
        score_entropy = self._rank_normalize(entropy)
        score_stability = self._rank_normalize(stability)
        score_topk = self._rank_normalize(topk_freq)
        score_depth = self._rank_normalize(depth, invert=True)  # Shallower is better
        
        # Geometry-specific weights (enhanced from De Prado with depth)
        if self.config.horizon_bars == 12:
            # Short-term: favor predictive power and early tree usage
            composite = (
                0.30 * score_importance + 
                0.20 * score_ic +
                0.15 * score_depth +      # NEW: Root proximity
                0.15 * score_topk +
                0.10 * score_entropy +
                0.10 * score_stability
            )
        else:
            # Long-term: favor stability and structural importance
            composite = (
                0.25 * score_importance + 
                0.15 * score_ic +
                0.20 * score_depth +      # NEW: Root proximity
                0.15 * score_topk +
                0.10 * score_entropy +
                0.15 * score_stability
            )
        
        return composite
    
    def sieve_3_4_enhanced_dominance_stability(self, X: pd.DataFrame, y: pd.Series, 
                                             sample_weight: Optional[pd.Series] = None) -> List[str]:
        """
        Enhanced Sieve 3/4 with De Prado predictive metrics and structural analysis.
        Now includes root proximity (tree depth) analysis.
        """
        tprint_info(f"🔍 Enhanced Sieve 3/4: LGBM + IC + Depth + Entropy + Stability ({len(X.columns)} features)")
        
        # Enhanced importance generation
        importance_results = self._generate_enhanced_lgbm_importance(X, y, sample_weight)
        mean_importance = importance_results['mean_importance']
        median_feature_ic = importance_results['median_feature_ic']
        oof_ic = importance_results['oof_ic']
        stability_scores = importance_results['stability']
        topk_freq = importance_results['topk_freq']
        depth_scores = importance_results['depth_scores']  # NEW: Root proximity
        
        # Shannon entropy (vectorized)
        entropy_scores = self._compute_shannon_entropy(X)
        
        # Enhanced composite scoring with depth
        composite_scores = self._compute_enhanced_composite_score(
            mean_importance, median_feature_ic, entropy_scores, stability_scores, topk_freq, depth_scores
        )
        
        # Create comprehensive feature stats (memory efficient)
        self.feature_stats_ = pd.DataFrame({
            'Importance': mean_importance,
            'IC': median_feature_ic,
            'Depth': depth_scores,  # NEW: Root proximity
            'Entropy': entropy_scores,
            'Stability': stability_scores,
            'TopKFreq': topk_freq,
            'CompositeScore': composite_scores
        })
        
        # Enhanced selection with multiple criteria
        # 1. Stability filter
        stable_mask = stability_scores <= self.config.instability_threshold
        stable_features = self.feature_stats_[stable_mask]
        
        # 2. Top-K frequency gate (from De Prado)
        topk_mask = topk_freq >= 0.3  # Features must be in top 20% at least 30% of the time
        gated_features = stable_features[topk_mask]
        
        # 3. Composite ranking within gated features
        if len(gated_features) > 0:
            ranked_features = gated_features.sort_values('CompositeScore', ascending=False)
            
            # Geometry-specific selection
            if self.config.horizon_bars == 12:
                # Short-term: more aggressive selection
                n_select = max(1, min(len(ranked_features), int(len(X.columns) * 0.3)))
            else:
                # Long-term: more conservative selection
                n_select = max(1, min(len(ranked_features), int(len(X.columns) * 0.4)))
            
            selected_features = ranked_features.head(n_select).index.tolist()
        else:
            # Fallback: top features by composite score
            ranked_all = self.feature_stats_.sort_values('CompositeScore', ascending=False)
            selected_features = ranked_all.head(max(1, len(X.columns) // 10)).index.tolist()
        
        # Enhanced logging with depth information
        tprint_info(f"   📊 Enhanced Metrics Summary:")
        tprint_info(f"      OOF IC (Model): {oof_ic:.4f}")
        tprint_info(f"      Importance Range: [{mean_importance.min():.4f}, {mean_importance.max():.4f}]")
        tprint_info(f"      IC Range: [{median_feature_ic.min():.4f}, {median_feature_ic.max():.4f}]")
        tprint_info(f"      Depth Range: [{depth_scores.min():.2f}, {depth_scores.max():.2f}]")  # NEW
        tprint_info(f"      Stability Threshold: {self.config.instability_threshold}")
        tprint_info(f"      Top-K Gate: {topk_mask.sum()} features passed")
        tprint_info(f"   📉 Enhanced reduction: {len(X.columns)} → {len(selected_features)} features")
        
        return selected_features

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None, 
                      use_enhanced: bool = True) -> pd.DataFrame:
        """
        Apply the complete 4-sieve pipeline to features.
        
        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Optional sample weights
            use_enhanced: Use enhanced De Prado metrics (default: True)
            
        Returns:
            Selected feature matrix
        """
        tprint_info(f"🚀 CausalFeatureSieve: {self.geometry} pipeline start (Enhanced: {use_enhanced})")
        tprint_info(f"📊 Input: {len(X.columns)} features, {len(X)} samples")
        
        # Detect regime features for special handling
        regime_features = self._detect_regime_features(X)
        if regime_features:
            tprint_info(f"🎭 Detected {len(regime_features)} regime features: {regime_features[:5]}...")
        
        initial_features = X.columns.tolist()
        T = len(X)
        
        # Sieve 1: ONC Clustering
        X_sieve1 = self.sieve_1_onc(X, T)
        if X_sieve1.empty:
            tprint_error("❌ Sieve 1 produced empty feature set")
            return X.iloc[:, :0]  # Return empty DataFrame
        
        # Sieve 2: ElasticNet + LASSO Selection
        selected_sieve2 = self.sieve_2_elastic_1se(X_sieve1, y)
        if not selected_sieve2:
            tprint_error("❌ Sieve 2 produced empty feature set")
            return X.iloc[:, :0]
        
        X_sieve2 = X_sieve1[selected_sieve2]
        
        # Sieve 3/4: Enhanced or Original LGBM + Stability
        if use_enhanced:
            selected_sieve4 = self.sieve_3_4_enhanced_dominance_stability(X_sieve2, y, sample_weight)
        else:
            selected_sieve4 = self.sieve_3_4_dominance_stability(X_sieve2, y, sample_weight)
            
        if not selected_sieve4:
            tprint_error("❌ Sieve 3/4 produced empty feature set")
            return X.iloc[:, :0]
        
        X_final = X_sieve2[selected_sieve4]
        
        # Summary
        reduction_rate = 1 - len(X_final.columns) / len(initial_features)
        method_name = "Enhanced" if use_enhanced else "Original"
        tprint_success(f"✅ CausalFeatureSieve ({method_name}): {self.geometry} complete!")
        tprint_success(f"📉 Feature reduction: {len(initial_features)} → {len(X_final.columns)} ({reduction_rate:.1%})")
        tprint_success(f"🎯 Final features: {X_final.columns.tolist()}")
        
        return X_final

def get_geometry_config(geometry: str) -> GeometryConfig:
    """Get geometry-specific configuration."""
    if geometry not in CausalFeatureSieve.GEOMETRY_CONFIGS:
        raise ValueError(f"Unknown geometry: {geometry}")
    return CausalFeatureSieve.GEOMETRY_CONFIGS[geometry]
