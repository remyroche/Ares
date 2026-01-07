import pandas as pd
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
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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
        Sieve 2: Parsimony via ElasticNet (1-SE Rule).
        Issue 4 Fix: Explicit CV awareness + Scale-within-Pipe.
        """
        regime_features = self._detect_regime_features(X)
        tprint_info(f"🔍 Sieve 2: ElasticNet Selection ({len(X.columns)} features)")
        if regime_features:
            tprint_info(f"   🎭 Processing {len(regime_features)} regime features with ElasticNet")
        
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
        selected_features = X.columns[coefs != 0].tolist()
        
        tprint_info(f"   🎯 Alpha rule: {self.config.alpha_rule}, alpha: {alpha_1se:.6f}")
        tprint_info(f"   📉 ElasticNet reduced: {len(X.columns)} → {len(selected_features)} features")
        
        return selected_features

    def _generate_mda_importance(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate MDA importance DataFrame using existing MDA_SHAP_FeatureSelector.
        """
        tprint_info(f"   📊 Generating MDA importance for {len(X.columns)} features")
        
        try:
            from src.training.steps.labeling.mda_shap_feature_selection import MDA_SHAP_FeatureSelector
            
            # Use fast settings for MDA generation
            mda_selector = MDA_SHAP_FeatureSelector(
                model_type="rf",
                n_folds=min(5, len(X) // 100),  # Adaptive folds
                embargo_pct=0.01,
                random_state=self.seed,
                verbose=False,  # Reduce verbosity
                subsample_train_pct=0.7,
                max_train_samples=5000,
                shap_max_evals=500  # Faster SHAP
            )
            
            # Generate MDA importance
            mda_results = mda_selector.fit_transform(
                X=X, 
                y=y, 
                target_sample_weight=sample_weight
            )
            
            # Extract per-feature MDA importance as DataFrame
            if hasattr(mda_selector, 'mda_results') and mda_selector.mda_results:
                # Convert MDA results to DataFrame format expected by sieve 3/4
                mda_importance_data = {}
                for feature, stats in mda_selector.mda_results.items():
                    if isinstance(stats, dict):
                        mda_importance_data[feature] = [stats.get('mean', 0.0)]
                    else:
                        mda_importance_data[feature] = [float(stats)]
                
                mda_importance_df = pd.DataFrame(mda_importance_data).T
                mda_importance_df.columns = ['importance']
                
                tprint_info(f"   ✅ MDA importance generated: {len(mda_importance_df)} features")
                return mda_importance_df
            else:
                tprint_warning("   ⚠️ MDA generation failed, using fallback importance")
                # Fallback: simple correlation-based importance
                importance_scores = {}
                for col in X.columns:
                    try:
                        corr = X[col].corr(y)
                        importance_scores[col] = [abs(corr) if not np.isnan(corr) else 0.0]
                    except:
                        importance_scores[col] = [0.0]
                
                fallback_df = pd.DataFrame(importance_scores).T
                fallback_df.columns = ['importance']
                return fallback_df
                
        except Exception as e:
            tprint_error(f"   ❌ MDA generation failed: {e}")
            # Fallback to simple importance
            importance_scores = {}
            for col in X.columns:
                try:
                    corr = X[col].corr(y)
                    importance_scores[col] = [abs(corr) if not np.isnan(corr) else 0.0]
                except:
                    importance_scores[col] = [0.0]
            
            fallback_df = pd.DataFrame(importance_scores).T
            fallback_df.columns = ['importance']
            return fallback_df

    def sieve_3_4_dominance_stability(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> List[str]:
        """
        Sieve 3 & 4: Dominance-Weighted Rank Stability.
        Issue 5 Fix: Penalize low importance even if stable.
        """
        tprint_info(f"🔍 Sieve 3/4: MDA + Stability Analysis ({len(X.columns)} features)")
        
        # Generate MDA importance DataFrame
        mda_importance_df = self._generate_mda_importance(X, y, sample_weight)
        
        if mda_importance_df.empty:
            tprint_warning("   ⚠️ No MDA importance available, returning all features")
            return X.columns.tolist()
        
        # Create multiple MDA runs for stability analysis
        n_runs = min(5, max(3, len(X) // 1000))  # Adaptive number of runs
        mda_runs = []
        
        for run in range(n_runs):
            # Add some randomness for stability testing
            np.random.seed(self.seed + run)
            run_importance = self._generate_mda_importance(X, y, sample_weight)
            if not run_importance.empty:
                mda_runs.append(run_importance)
        
        if len(mda_runs) < 2:
            tprint_warning("   ⚠️ Insufficient MDA runs for stability analysis")
            # Use single run results
            mean_importance = mda_importance_df.iloc[:, 0]
            std_rank = pd.Series(0, index=mda_importance_df.index)
            instability = std_rank / (mean_importance.rank() + 1e-9)
        else:
            # Combine multiple runs
            combined_importance = pd.concat(mda_runs, axis=1)
            mean_importance = combined_importance.mean(axis=1)
            
            # Calculate rank stability across runs
            ranks = combined_importance.rank(axis=0, ascending=False, method='min')
            mean_rank = ranks.mean(axis=1)
            std_rank = ranks.std(axis=1)
            
            # Instability Index
            instability = std_rank / (mean_rank + 1e-9)
        
        # Issue 5 Fix: Dominance weighting
        dominance_stability_score = mean_importance / (instability + 1e-9)
        
        results = pd.DataFrame({
            'mean_importance': mean_importance,
            'instability_index': instability,
            'dom_stab_score': dominance_stability_score,
            'is_stable': instability <= self.config.instability_threshold
        }).sort_values('dom_stab_score', ascending=False)
        
        stable_features = results[results['is_stable']].index.tolist()
        
        tprint_info(f"   📊 MDA runs: {n_runs}, stable threshold: {self.config.instability_threshold}")
        tprint_info(f"   📉 Stability reduced: {len(X.columns)} → {len(stable_features)} features")
        tprint_info(f"   📈 Stability rate: {len(stable_features)/len(X.columns):.1%}")
        
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
            'volatility_regime'
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
        
        # Sieve 2: ElasticNet Selection
        selected_sieve2 = self.sieve_2_elastic_1se(X_sieve1, y)
        if not selected_sieve2:
            tprint_error("❌ Sieve 2 produced empty feature set")
            return X.iloc[:, :0]
        
        X_sieve2 = X_sieve1[selected_sieve2]
        
        # Sieve 3/4: MDA + Stability
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

def get_geometry_config(geometry: str) -> GeometryConfig:
    """Get geometry-specific configuration."""
    if geometry not in CausalFeatureSieve.GEOMETRY_CONFIGS:
        raise ValueError(f"Unknown geometry: {geometry}")
    return CausalFeatureSieve.GEOMETRY_CONFIGS[geometry]
