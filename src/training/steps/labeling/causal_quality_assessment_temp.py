import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
except ImportError:
    # Fallback to standard logging if tprint not available
    def tprint_info(msg): logger.info(msg)
    def tprint_warning(msg): logger.warning(msg)
    def tprint_error(msg): logger.error(msg)
    def tprint_success(msg): logger.info(f"✅ {msg}")

logger = logging.getLogger(__name__)

class CausalQualityAssessor:
    """
    Implements a rigorous De Prado-aligned metric stack for assessing causal discovery quality
    before passing events to downstream layers.
    
    Metrics Groups:
    1. Causal Validity (Structure-Level)
    2. Temporal Stability
    3. Predictive Integrity
    4. Multiple-Testing Robustness
    5. Complexity & Parsimony
    """
    
    # Hard survival filters (must pass ALL before Layer2Score computation)
    SURVIVAL_FILTERS = {
        'CI_score': (0.05, float('inf'), 'Conditional independence too weak'),
        'PSR': (0.5, 0.9, 'Parent sufficiency out of range'),
        'CV_freq': (0.0, 0.60, 'Event frequency too unstable'),
        'IR_cv': (0.0, 0.8, 'Impact stability too volatile'),
        'Dir_consistency': (0.40, 1.0, 'Direction flips too frequent'),
        'OOS_R2': (0.03, 1.0, 'Predictive power insufficient'),
        'IC': (0.1, 1.0, 'Information coefficient too weak'),
        'IC_IR': (0.5, float('inf'), 'IC stability insufficient'),
        'DSR': (0.5, 1.0, 'Deflated Sharpe Ratio too low'),
    }
    MIN_EVENTS_SURVIVAL = 50
    
    def __init__(self, verbose: bool = False, enable_survival_filters: bool = True):
        self.verbose = verbose
        self.enable_survival_filters = enable_survival_filters
        self.survival_failures = {}  # Track why candidates failed
        
    def assess_candidate(self, 
                         candidate: Any, 
                         df: pd.DataFrame, 
                         events_df: pd.DataFrame, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         backbone_features: Optional[pd.DataFrame] = None,
                         precomputed_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Run full assessment suite on a causal candidate.
        """
        candidate_id = getattr(candidate, 'uuid', 'unknown')[:8]
        if self.verbose:
            tprint_info(f"🔍 Starting quality assessment for candidate {candidate_id}")
        
        # Enhanced data alignment and validation
        alignment_result = self._validate_and_align_data(candidate_id, df, events_df, X, y)
        if not alignment_result['valid']:
            error_msg = alignment_result.get('error', 'Unknown error')
            if self.verbose:
                tprint_error(f"❌ Candidate {candidate_id}: Data validation failed - {error_msg}")
            return self._get_default_metrics()
        
        events_df, X, y = alignment_result['events_df'], alignment_result['X'], alignment_result['y']
        
        # ========== EARLY BACKBONE REDUNDANCY CHECK ==========
        # Prune candidates that are just proxies for existing Specialists
        if backbone_features is not None and not backbone_features.empty:
            is_redundant, reason = self._check_backbone_redundancy(X, backbone_features)
            if is_redundant:
                if self.verbose:
                    tprint_warning(f"⚠️ Candidate {candidate_id} PRUNED: {reason}")
                
                # Fail immediately
                m = self._get_default_metrics()
                m['survival_status'] = 'FAILED'
                self.survival_failures[candidate_id] = [reason]
                return m

        
        # ========== NEW: GLOBAL ITERATIVE FEATURE SELECTION (De Prado-aligned) ==========
        # Reduce 556+ features to ~100 (or fewer for small samples) once for ALL downstream assessment steps
        target_n_features = min(100, max(10, int(len(y) / 5)))
        
        if precomputed_features is not None:
             # Use shared family features (Optimization #3)
             valid_feats = [f for f in precomputed_features if f in X.columns]
             if len(valid_feats) > 0:
                 if self.verbose:
                     tprint_info(f"   ⏩ Using {len(valid_feats)} precomputed family features")
                 X = X[valid_feats]
        elif X.shape[1] > target_n_features:
            if self.verbose:
                tprint_info(f"   🌲 Pre-selection: Reducing {X.shape[1]} features to {target_n_features} via iterative LightGBM...")
            X_selected = self._perform_iterative_selection(X, y, target_features=target_n_features)
            X = X_selected
            
        # Attach selected features to candidate for caching by caller
        if hasattr(candidate, 'selected_features'):
            candidate.selected_features = list(X.columns)
        elif isinstance(candidate, dict):
            candidate['selected_features'] = list(X.columns)
                
        if self.verbose:
            tprint_info(f"   📊 Candidate {candidate_id}: {len(events_df)} events, {X.shape[1]} features, target range [{y.min():.4f}, {y.max():.4f}]")
        
        metrics = {}
        
        # 1. Validity (Uses downsampled X + Backbone Context)
        metrics.update(self.compute_validity_metrics(candidate, X, y, backbone_features=backbone_features))
        
        # 2. Stability
        metrics.update(self.compute_stability_metrics(events_df, y))
        
        # 3. Predictive Integrity (Uses downsampled X)
        metrics.update(self.compute_predictive_integrity(X, y))
        
        # 4. Robustness
        metrics.update(self.compute_robustness_metrics(y))
        
        # 5. Complexity
        metrics.update(self.compute_complexity_metrics(candidate, events_df))
        
        # 6. Causal Specifics
        metrics['Parent_Overlap'] = metrics.get('Overlap_Ratio', 0.0)
        metrics['Interventional_Contrast'] = metrics.get('CI_score', 0.0) * metrics.get('Dir_consistency', 0.5)
        metrics['Overlap_Support'] = 1.0 - metrics.get('Overlap_Ratio', 0.0)
        metrics['Path_Stability'] = metrics.get('IR_cv', 1.0)
        metrics['Structural_Importance'] = metrics.get('CI_score', 0.0) * (1.0 + metrics.get('IC', 0.0))
        
        # 7. Apply Survival Filters (BEFORE composite score)
        if self.enable_survival_filters:
            family = getattr(candidate, 'family', None)
            if not family and isinstance(candidate, dict):
                family = candidate.get('family')
                
            passed_filters, filter_failures = self._apply_survival_filters(metrics, len(events_df), family=family)
            
            if not passed_filters:
                if self.verbose:
                    tprint_warning(f"⚠️ Candidate {candidate_id} FAILED survival filters: {filter_failures}")
                self.survival_failures[candidate_id] = filter_failures
                metrics['Layer2Score'] = 0.0
                metrics['survival_status'] = 'FAILED'
            else:
                metrics['Layer2Score'] = self.compute_composite_score(metrics)
                metrics['survival_status'] = 'PASSED'
                if self.verbose:
                    tprint_success(f"✅ Candidate {candidate_id} PASSED survival filters")
        else:
            metrics['Layer2Score'] = self.compute_composite_score(metrics)
            metrics['survival_status'] = 'NO_FILTER'
        
        if self.verbose:
            tprint_success(f"✅ Candidate {candidate_id} assessment complete (Layer2Score: {metrics.get('Layer2Score', 0.0):.4f})")
        return metrics

    def _validate_and_align_data(self, candidate_id: str, df: pd.DataFrame, events_df: pd.DataFrame, 
                                X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        try:
            if not isinstance(events_df.index, pd.DatetimeIndex):
                events_df.index = pd.to_datetime(events_df.index)
            if not isinstance(y.index, pd.DatetimeIndex):
                y.index = pd.to_datetime(y.index)
            if not isinstance(X.index, pd.DatetimeIndex):
                X.index = pd.to_datetime(X.index)
            
            if len(events_df) == 0: return {'valid': False, 'error': 'Events empty'}
            if len(X) == 0: return {'valid': False, 'error': 'X empty'}
            if len(y) == 0: return {'valid': False, 'error': 'y empty'}
            
            common_index = events_df.index.intersection(X.index).intersection(y.index)
            if len(common_index) < 10: return {'valid': False, 'error': f'Aligned samples {len(common_index)} < 10'}
            
            X_aligned = X.loc[common_index]
            y_aligned = y.loc[common_index]
            valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
            if valid_mask.sum() < 10: return {'valid': False, 'error': f'Valid samples after NaN removal {valid_mask.sum()} < 10'}
            
            X_clean = X_aligned[valid_mask]
            constant_features = X_clean.nunique() == 1
            if constant_features.any():
                X_clean = X_clean.loc[:, ~constant_features]
            
            if X_clean.shape[1] == 0: return {'valid': False, 'error': 'No valid features'}
            if y_aligned[valid_mask].var() == 0: return {'valid': False, 'error': 'Target zero variance'}
            
            return {
                'valid': True,
                'events_df': events_df.loc[common_index][valid_mask],
                'X': X_clean,
                'y': y_aligned[valid_mask]
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def _apply_survival_filters(self, metrics: Dict[str, float], events_count: int, family: str = None) -> Tuple[bool, List[str]]:
        failures = []
        if events_count < self.MIN_EVENTS_SURVIVAL:
            failures.append(f"min_events={events_count} < {self.MIN_EVENTS_SURVIVAL}")
        
        for metric_name, (min_val, max_val, reason) in self.SURVIVAL_FILTERS.items():
            value = metrics.get(metric_name, 0.0)
            current_min = min_val
            if metric_name == 'OOS_R2' and family and 'COMPOSITE' in str(family):
                current_min = 0.0
            if not (current_min <= value <= max_val):
                failures.append(f"{metric_name}={value:.4f} not in [{current_min}, {max_val}] - {reason}")
        
        return len(failures) == 0, failures

    def _check_backbone_redundancy(self, X: pd.DataFrame, backbone: pd.DataFrame, threshold: float = 0.95) -> Tuple[bool, str]:
        """
        Check if any candidate feature is highly correlated with any backbone feature.
        """
        try:
            # Align indices
            common_idx = X.index.intersection(backbone.index)
            if len(common_idx) < 50:
                return False, ""
                
            X_aligned = X.loc[common_idx]
            back_aligned = backbone.loc[common_idx]
            
            # Use correlation on a subset for speed if X is huge
            # We check the first 20 features (typically the most important ones)
            if X_aligned.shape[1] > 20:
                X_check = X_aligned.iloc[:, :20] 
            else:
                X_check = X_aligned

            # Compute correlation matrix efficiently using numpy
            # Standardize first
            X_std = (X_check - X_check.mean()) / (X_check.std() + 1e-9)
            B_std = (back_aligned - back_aligned.mean()) / (back_aligned.std() + 1e-9)
            
            # Covariance matrix (Correlation since standardized)
            # Result: (N_X x N_B)
            corr_mat = np.dot(X_std.T, B_std) / (len(common_idx) - 1)
            corr_mat = np.abs(corr_mat)
            
            max_corr = np.max(corr_mat)
            
            if max_corr > threshold:
                # Find which pairwise correlation violated the threshold
                idx = np.unravel_index(np.argmax(corr_mat), corr_mat.shape)
                feat_x = X_check.columns[idx[0]]
                feat_b = back_aligned.columns[idx[1]]
                return True, f"High correlation ({max_corr:.4f}) with backbone: {feat_x} vs {feat_b}"
                
            return False, ""
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Redundancy check failed: {e}")
            return False, ""

    def _get_default_metrics(self) -> Dict[str, float]:
        m = {k: 0.0 for k in self.SURVIVAL_FILTERS.keys()}
        m.update({
            'Layer2Score': 0.0, 'survival_status': 'FAILED',
            'Parent_Overlap': 0.0, 'Interventional_Contrast': 0.0,
            'Overlap_Support': 0.0, 'Path_Stability': 0.0, 'Structural_Importance': 0.0,
            'Sparsity': 3.0, 'Overlap_Ratio': 0.0, 'SPA_p': 1.0
        })
        return m

    def _perform_iterative_selection(self, X: pd.DataFrame, y: pd.Series, target_features: int = 100) -> pd.DataFrame:
        """
        Helper to perform iterative LightGBM feature selection (70% Gain + 30% Split + 20% Depth Decay).
        """
        import time
        start_time = time.time()
        
        try:
            # ========== STEP 1: MI-PROXY DOWNSAMPLING (556 → 300) ==========
            MAX_FEATURES_MI = 300
            if X.shape[1] > MAX_FEATURES_MI:
                try:
                    # Use correlation with target as MI proxy (fast)
                    correlations = X.corrwith(y).abs().fillna(0)
                    top_features = correlations.nlargest(MAX_FEATURES_MI).index
                    X = X[top_features]
                except Exception: pass
            
            # ========== STEP 2: ITERATIVE LGBM FEATURE SUBSETTING ==========
            TARGET_FEATURES = target_features
            iteration = 0
            max_iterations = 5  # Increased to ensure we reach target
            
            while X.shape[1] > TARGET_FEATURES and iteration < max_iterations:
                iteration += 1
                n_features = X.shape[1]
                n_subsets = min(3, max(1, n_features // 100))
                subset_size = n_features // n_subsets
                
                all_importances = {}
                feature_list = list(X.columns)
                
                for subset_idx in range(n_subsets):
                    start_idx = subset_idx * subset_size
                    end_idx = min((subset_idx + 1) * subset_size, n_features)
                    subset_features = feature_list[start_idx:end_idx]
                    X_subset = X[subset_features]
                    
                    try:
                        import lightgbm as lgb
                        # Check labels - if constant, skip fit and use default importance
                        y_binary = (y > y.median()).astype(int) if y.dtype == float else y
                        if len(np.unique(y_binary)) < 2:
                            for feat in subset_features:
                                all_importances[feat] = 0.5
                            continue

                        model = lgb.LGBMClassifier(
                            n_estimators=30, max_depth=3, num_leaves=8,
                            learning_rate=0.1, verbosity=-1, n_jobs=1, # Single job to avoid hangs
                            random_state=42
                        )
                        model.fit(X_subset, y_binary)
                        
                        booster = model.booster_
                        gain_imp = model.feature_importances_
                        split_imp = booster.feature_importance(importance_type='split')
                        
                        # Depth decay factor (0.8^avg_depth)
                        depth_decay = np.ones(len(subset_features))
                        try:
                            trees_df = booster.trees_to_dataframe()
                            if 'split_feature' in trees_df.columns:
                                split_nodes = trees_df[trees_df['split_feature'].notna()]
                                depth_sums = np.zeros(len(subset_features))
                                depth_counts = np.zeros(len(subset_features))
                                for _, row in split_nodes.iterrows():
                                    feat = row.get('split_feature', None)
                                    if feat in subset_features:
                                        idx = subset_features.index(feat)
                                        depth = int(row.get('node_depth', 0))
                                        depth_sums[idx] += depth
                                        depth_counts[idx] += 1
                                for i in range(len(subset_features)):
                                    if depth_counts[i] > 0:
                                        depth_decay[i] = 0.8 ** (depth_sums[i] / depth_counts[i])
                        except Exception: pass
                        
                        # Identify backbone features to protect them (Specialists and Regimes)
                        backbone_prefixes = ['SPECIALIST', 'REGIME', '_PC1', '_PC2', '_PC3', 'rv_z_short']
                        
                        gain_norm = gain_imp / (gain_imp.max() + 1e-8)
                        split_norm = split_imp / (split_imp.max() + 1e-8)
                        composite = (0.70 * gain_norm + 0.30 * split_norm) * depth_decay
                        
                        for i, feat in enumerate(subset_features):
                            is_backbone = any(p in feat for p in backbone_prefixes)
                            score = composite[i]
                            
                            # Protect backbone: even if weak, don't let it drop too easily
                            # Dampen importance (0.3x) so it doesn't block top signals, but we'll force-keep it
                            if is_backbone:
                                all_importances[feat] = max(0.4, score * 0.3) 
                            else:
                                all_importances[feat] = score
                    except Exception:
                        for feat in subset_features: all_importances[feat] = 0.5
                
                if not all_importances: break
                
                keep_ratio = 0.75
                sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)
                
                # FORCE KEEP Backbone features
                backbone_prefixes = ['SPECIALIST', 'REGIME', '_PC1', '_PC2', '_PC3', 'rv_z_short']
                must_keep = [f for f in X.columns if any(p in f for p in backbone_prefixes)]
                
                n_keep = max(TARGET_FEATURES, int(len(sorted_features) * keep_ratio))
                kept_features = [f for f, _ in sorted_features[:n_keep]]
                
                # Ensure all must_keep are in kept_features
                final_keep = list(set(kept_features) | set(must_keep))
                X = X[final_keep]
                
            return X
        except Exception:
            return X[:100] if X.shape[1] > 100 else X

    def compute_validity_metrics(self, 
                               candidate, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               backbone_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute validity metrics using LightGBM with composite importance scoring.
        Injects backbone features (context) to ensure Fair OOS R2 assessment.
        """
        import time
        start_time = time.time()
        
        try:
            # Inject backbone context if available (critical for Regime-Conditional signals)
            X_eval = X.copy()
            if backbone_features is not None and not backbone_features.empty:
                # Align indices first
                common_idx = X.index.intersection(backbone_features.index)
                if len(common_idx) > 30:
                    # Select top 10 backbone features to avoid dim explosion
                    bb_subset = backbone_features.loc[common_idx].iloc[:, :10]
                    X_eval = pd.concat([X.loc[common_idx], bb_subset], axis=1)
                    y = y.loc[common_idx]
                
            if self.verbose:
                tprint_info(f"   🔬 Computing validity metrics for {X.shape[1]} features + {X_eval.shape[1]-X.shape[1]} backbone context")
            
            # ========== STEP 3: COMPUTE METRICS WITH FINAL FEATURES (X_eval) ==========
            if len(y) > 30:
                n_splits = 3 if len(y) > 500 else 2
                cv = TimeSeriesSplit(n_splits=n_splits)
                cv_iterator = cv.split(X_eval)
            else:
                # Fallback
                split_idx = int(len(y) * 0.7)
                cv_iterator = [(list(range(split_idx)), list(range(split_idx, len(y))))]
            
            r2_scores = []
            
            # Use Ridge for final R² computation
            ridge_solver = 'lsqr'
            for train_idx, val_idx in cv_iterator:
                if len(train_idx) < 10: continue
                model = Ridge(alpha=1.0, solver=ridge_solver)
                model.fit(X_eval.iloc[train_idx], y.iloc[train_idx])
                # Score on X_eval (candidate + backbone)
                r2 = model.score(X_eval.iloc[val_idx], y.iloc[val_idx])
                r2_scores.append(max(0.0, r2))
            ci_score = np.mean(r2_scores) if r2_scores else 0.0
            
            # PSR with reduced bootstrap (Use original X for feature stability)
            # We want to know if the CANDIDATE features are stable, not the backbone
            full_model = Ridge(alpha=1.0, solver=ridge_solver)
            full_model.fit(X, y)
            
            n_bootstrap = 3  # Minimal for speed
            importances = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(X), size=len(X), replace=True)
                boot_model = Ridge(alpha=1.0, solver=ridge_solver)
                boot_model.fit(X.iloc[idx], y.iloc[idx])
                importances.append(np.abs(boot_model.coef_))
            
            if importances:
                mean_imp = np.mean(importances, axis=0)
                std_imp = np.std(importances, axis=0)
                denom = np.where(mean_imp > 1e-6, mean_imp, 1e-6)
                feat_stab = 1.0 - np.mean(std_imp / denom)
            else:
                feat_stab = 0.5
            
            full_residuals = y - full_model.predict(X)
            res_series = pd.Series(full_residuals)
            res_autocorr = np.abs(res_series.autocorr()) if len(res_series) > 10 else 0.5
            if np.isnan(res_autocorr): res_autocorr = 0.5
            
            psr = 0.6 * max(0, min(1, feat_stab)) + 0.4 * (1.0 - res_autocorr)
            
            elapsed = time.time() - start_time
            if self.verbose:
                tprint_info(f"   ✅ CI_score: {ci_score:.4f}, PSR: {psr:.4f} (total: {elapsed:.2f}s)")
            return {'CI_score': ci_score, 'PSR': psr}
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Validity failed: {e}")
            return {'CI_score': 0.0, 'PSR': 0.0}


    def compute_stability_metrics(self, events_df: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        try:
            if self.verbose:
                tprint_info(f"   📈 Computing stability metrics...")
            time_span = events_df.index.max() - events_df.index.min()
            freq = 'W' if time_span.days >= 7 else 'D'
            counts = events_df.resample(freq).size()
            cv_freq = counts.std() / counts.mean() if counts.mean() > 0 else 10.0
            
            window = max(5, len(y) // 5)
            rolling_returns = y.rolling(window)
            r_std = rolling_returns.std()
            r_mean = rolling_returns.mean()
            rolling_ir = r_mean / (r_std + 1e-9)
            valid_ir = rolling_ir.dropna()
            
            if len(valid_ir) >= 3:
                ir_mean = valid_ir.mean()
                ir_std = valid_ir.std()
                ir_cv = abs(ir_std / (ir_mean + 1e-9)) if abs(ir_mean) > 1e-6 else 10.0
            else:
                ir_cv = 10.0
            
            consistencies = []
            y_mean_sign = np.sign(y.mean())
            for w in [max(5, len(y)//10), max(3, len(y)//15)]:
                if w >= len(y): continue
                rolling_mean = y.rolling(w).mean()
                consistencies.append((np.sign(rolling_mean) == y_mean_sign).dropna().mean())
            dir_stab = np.mean(consistencies) if consistencies else 0.0
            
            if self.verbose:
                tprint_info(f"   ✅ CV_freq: {cv_freq:.4f}, IR_cv: {ir_cv:.4f}, Dir_consistency: {dir_stab:.4f}")
            return {'CV_freq': cv_freq, 'IR_cv': ir_cv, 'Dir_consistency': dir_stab}
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Stability failed: {e}")
            return {'CV_freq': 10.0, 'IR_cv': 10.0, 'Dir_consistency': 0.0}

    def compute_predictive_integrity(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        try:
            if self.verbose:
                tprint_info(f"   🎯 Computing predictive integrity...")
            n_splits = 2 if len(y) > 2000 else 3
            tscv = TimeSeriesSplit(n_splits=n_splits)
            ridge_solver = 'lsqr' if X.shape[1] > 100 else 'auto'
            oos_r2_scores = []
            for train_idx, test_idx in tscv.split(X):
                if len(train_idx) < 10: continue
                model = Ridge(alpha=1.0, solver=ridge_solver)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                oos_r2_scores.append(max(0.0, model.score(X.iloc[test_idx], y.iloc[test_idx])))
            oos_r2 = np.mean(oos_r2_scores) if oos_r2_scores else 0.0
            
            corrs = X.corrwith(y).abs()
            ic = corrs.max() if not corrs.empty else 0.0
            best_feat = corrs.idxmax() if not corrs.empty else None
            ic_ir = 0.0
            if best_feat:
                window = max(5, len(y) // 5)
                rolling_ic = X[best_feat].rolling(window).corr(y).dropna()
                if len(rolling_ic) >= 3:
                    ic_ir_mean = rolling_ic.mean()
                    ic_ir_std = rolling_ic.std()
                    ic_ir = abs(ic_ir_mean / (ic_ir_std + 1e-9))
            
            if self.verbose:
                tprint_info(f"   ✅ OOS_R2: {oos_r2:.4f}, IC: {ic:.4f}, IC_IR: {ic_ir:.4f}")
            return {'OOS_R2': oos_r2, 'IC': ic, 'IC_IR': ic_ir}
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Integrity failed: {e}")
            return {'OOS_R2': 0.0, 'IC': 0.0, 'IC_IR': 0.0}

    def compute_robustness_metrics(self, y: pd.Series) -> Dict[str, float]:
        try:
            if self.verbose:
                tprint_info(f"   🛡️ Computing robustness metrics...")
            r = np.asarray(y)
            n = len(r)
            sr = r.mean() / (r.std() + 1e-9)
            skew = stats.skew(r)
            kurt = stats.kurtosis(r, fisher=False)
            
            denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
            if denom <= 0: z = 0.0
            else: z = sr * np.sqrt(n - 1) / denom
            
            dsr = stats.norm.cdf(z - stats.norm.ppf(1 - 1/100))
            
            r_null = r - r.mean()
            boot_sr = []
            for _ in range(250):
                sample = np.random.choice(r_null, size=n, replace=True)
                s_std = sample.std()
                if s_std > 1e-9:
                    boot_sr.append(sample.mean() / s_std)
            
            spa_p = np.mean(np.array(boot_sr) >= sr) if boot_sr else 1.0
            
            if self.verbose:
                tprint_info(f"   ✅ DSR: {dsr:.4f}, SPA_p: {spa_p:.4f}")
            return {'DSR': float(dsr), 'SPA_p': float(spa_p)}
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Robustness failed: {e}")
            return {'DSR': 0.0, 'SPA_p': 1.0}

    def compute_complexity_metrics(self, candidate, events_df: pd.DataFrame) -> Dict[str, float]:
        try:
            if self.verbose:
                tprint_info(f"   🧮 Computing complexity metrics...")
            
            horizon = 12
            if hasattr(candidate, 'params') and isinstance(candidate.params, dict):
                horizon = candidate.params.get('horizon', 12)
            elif isinstance(candidate, dict) and 'params' in candidate:
                horizon = candidate['params'].get('horizon', 12)
            
            events_sorted = events_df.sort_index()
            end_times = events_sorted.index + pd.Timedelta(minutes=15 * horizon)
            overlaps = 0
            if len(events_sorted) > 0:
                latest_end = end_times[0]
                for i in range(1, len(events_sorted)):
                    if events_sorted.index[i] < latest_end:
                        overlaps += 1
                    latest_end = max(latest_end, end_times[i])
            overlap_ratio = overlaps / len(events_df) if len(events_df) > 0 else 0.0
            
            selected_features = getattr(candidate, 'selected_features', [])
            n_feats = len(selected_features) if selected_features is not None else 0
            sparsity = min(10.0, n_feats / 5.0) if n_feats > 0 else 3.0
            
            if self.verbose:
                tprint_info(f"   ✅ Sparsity: {sparsity:.4f}, Overlap_Ratio: {overlap_ratio:.4f}")
            return {'Sparsity': sparsity, 'Overlap_Ratio': overlap_ratio}
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Complexity failed: {e}")
            return {'Sparsity': 3.0, 'Overlap_Ratio': 0.0}

    def compute_composite_score(self, metrics: Dict[str, float]) -> float:
        try:
            val_score = 0.6 * max(0, min(1, metrics.get('CI_score', 0.0))) + 0.4 * max(0, min(1, metrics.get('PSR', 0.0)))
            stab_score = np.mean([
                1.0 / (1.0 + max(0, metrics.get('CV_freq', 10.0) - 0.3)),
                1.0 / (1.0 + max(0, metrics.get('IR_cv', 10.0) - 0.5)),
                max(0, min(1, metrics.get('Dir_consistency', 0.0)))
            ])
            integ_score = np.mean([
                min(1.0, max(0, metrics.get('OOS_R2', 0.0)) / 0.1),
                min(1.0, abs(metrics.get('IC', 0.0)) / 0.05),
                min(1.0, metrics.get('IC_IR', 0.0) / 1.0)
            ])
            rob_score = np.mean([max(0, min(1, metrics.get('DSR', 0.0))), 1.0 - metrics.get('SPA_p', 1.0)])
            overlap = metrics.get('Overlap_Ratio', 0.0)
            sparsity = metrics.get('Sparsity', 3.0)
            comp_score = np.mean([
                1.0 if overlap <= 0.2 else max(0.0, 1.0 - (overlap - 0.2) * 2),
                1.0 if sparsity <= 2.0 else max(0.0, 1.0 - (sparsity - 2.0) * 0.2)
            ])
            
            final_score = 0.2 * (val_score + stab_score + integ_score + rob_score + comp_score)
            return float(max(0.0, min(1.0, final_score)))
        except Exception: return 0.0

    def _compute_deflated_sharpe_ratio(self, returns: pd.Series, n_trials: int = 100) -> float:
        try:
            r = np.asarray(returns)
            if len(r) < 2: return 0.0
            sr = r.mean() / (r.std() + 1e-9)
            skew = stats.skew(r)
            kurt = stats.kurtosis(r, fisher=False)
            denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
            if denom <= 0: z = 0.0
            else: z = sr * np.sqrt(len(r) - 1) / denom
            return float(stats.norm.cdf(z - stats.norm.ppf(1 - 1/n_trials)))
        except Exception: return 0.0

    def _compute_spa_test(self, returns: pd.Series, n_bootstrap: int = 500) -> float:
        try:
            r = np.asarray(returns)
            if len(r) < 10: return 1.0
            actual_sr = r.mean() / (r.std() + 1e-9)
            r_null = r - r.mean()
            boot_sr = []
            n = len(r)
            for _ in range(n_bootstrap):
                sample = np.random.choice(r_null, size=n, replace=True)
                s_std = sample.std()
                if s_std > 1e-9: boot_sr.append(sample.mean() / s_std)
            return float(np.mean(np.array(boot_sr) >= actual_sr) if boot_sr else 1.0)
        except Exception: return 1.0
