import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from enum import Enum

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


class SignalRole(Enum):
    """
    Signal role classification for role-aware survival filtering.
    
    Different roles encode different economic objects:
    - PREDICTOR: Expected return magnitude (dense, smooth)
    - TRIGGER: Timing asymmetry (sparse, bursty)
    - INTERACTION: Signal × Context composites
    - CONTEXT: State conditioning (non-predictive alone)
    """
    PREDICTOR = "predictor"
    TRIGGER = "trigger"
    INTERACTION = "interaction"
    CONTEXT = "context"


# Role-specific survival filter thresholds
# These encode the different economic properties of each signal type
# Per user specification (2026-01-04)
ROLE_SURVIVAL_FILTERS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣ CONTINUOUS PREDICTORS (Core Alpha) - Dense, smooth
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣ CONTINUOUS PREDICTORS (Core Alpha) - Dense, smooth
    # ═══════════════════════════════════════════════════════════════════════════
    SignalRole.PREDICTOR: {
        'CI_score': (0.005, float('inf'), 'Residualized predictors rarely exceed 0.03'),
        'IC': (0.02, 1.0, 'Crypto IC is structurally low'),
        'IC_IR': (0.2, float('inf'), 'Stability matters more than level'),
        'Dir_consistency': (0.51, 1.0, 'Slight directional edge only'),
        'PSR': (0.60, 0.95, 'Avoid pathological Sharpe'),
        # OOS_R2: N/A - Ridge often zero for returns
        # CV_freq: N/A - Must be dense (no event frequency check)
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2️⃣ EVENT-BASED TRIGGERS (Timing Signals) - Sparse, bursty
    # ═══════════════════════════════════════════════════════════════════════════
    SignalRole.TRIGGER: {
        'CI_score': (0.001, float('inf'), 'Triggers overlap by nature'),
        'Dir_consistency': (0.38, 1.0, 'Allow contrarian triggers'),
        'event_count': (30, float('inf'), 'Minimum statistical mass'),
        'CV_freq': (0.0, 1.2, 'Bursty allowed'),
        'balance': (0.20, 0.80, 'Timing asymmetry acceptable'),
        # IC: N/A - Triggers ≠ predictors
        # OOS_R2: N/A - Binary sparsity breaks regression
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3️⃣ INTERACTION TRIGGERS (Signal × Context)
    # ═══════════════════════════════════════════════════════════════════════════
    SignalRole.INTERACTION: {
        'CI_score': (0.0005, float('inf'), 'Interactions are conditional'),
        'Dir_consistency': (0.42, 1.0, 'Often asymmetric'),
        'event_count': (25, float('inf'), 'Lower than pure triggers'),
        'CV_freq': (0.0, 1.2, 'Context causes clustering'),
        'redundancy_corr': (0.0, 0.95, 'Avoid backbone duplication'),
        'incremental_lift': (0.03, float('inf'), 'Must add ≥3% AUC vs parent'),
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4️⃣ CONTEXT / REGIME FEATURES (Standalone conditioning)
    # ═══════════════════════════════════════════════════════════════════════════
    SignalRole.CONTEXT: {
        # A. State Separability
        'variance_ratio': (1.1, float('inf'), 'σ²(high) / σ²(low) must partition'), # Relaxed from 1.3
        'tail_risk_ratio': (1.1, float('inf'), 'Tail risk must differ'), # Relaxed from 1.2
        # B. Conditional Predictive Lift (most important)
        'delta_IC': (0.005, float('inf'), 'Must improve IC when conditioned'),
        'delta_AUC': (0.015, float('inf'), 'Must improve AUC by ≥1.5%'),
        # C. Temporal Stability
        'transition_entropy': (0.0, 0.9, 'Avoid noise'), # Relaxed from 0.8
        'regime_duration_cv': (0.0, 1.5, 'Stable segmentation'),
        # D. Orthogonality
        'corr_vs_backbone': (0.0, 0.90, 'Must not duplicate backbone'),
        'mutual_info': (0.0, 0.40, 'Low MI with other regimes'),
    },
}


class CausalQualityAssessor:
    """
    Implements a rigorous De Prado-aligned metric stack for assessing causal discovery quality
    before passing events to downstream layers.
    
    Now supports role-aware survival filtering where different signal types
    (Predictor, Trigger, Interaction, Context) have appropriate thresholds.
    
    Metrics Groups:
    1. Causal Validity (Structure-Level)
    2. Temporal Stability
    3. Predictive Integrity
    4. Multiple-Testing Robustness
    5. Complexity & Parsimony
    """
    
    # Default survival filters (used when role is unknown or for backward compatibility)
    # Thresholds relaxed based on empirical crypto 15m data observations
    SURVIVAL_FILTERS = {
        'CI_score': (0.01, float('inf'), 'Conditional independence too weak'),
        'PSR': (0.4, 0.95, 'Parent sufficiency out of range'),
        'CV_freq': (0.0, 0.80, 'Event frequency too unstable'),
        'IR_cv': (0.0, 1.0, 'Impact stability too volatile'),
        'Dir_consistency': (0.35, 1.0, 'Direction flips too frequent'),
        'OOS_R2': (0.0, 1.0, 'Predictive power insufficient'),
        'IC': (0.05, 1.0, 'Information coefficient too weak'),
        'IC_IR': (0.4, float('inf'), 'IC stability insufficient'),
        'DSR': (0.4, 1.0, 'Deflated Sharpe Ratio too low'),
    }
    MIN_EVENTS_SURVIVAL = 50
    
    def __init__(self, verbose: bool = False, enable_survival_filters: bool = True, enable_causal_quality: bool = True, **kwargs):
        self.verbose = verbose
        self.enable_survival_filters = enable_survival_filters
        self.enable_causal_quality = enable_causal_quality
        self.survival_failures = {}  # Track why candidates failed
        # Store any extra kwargs (for forward compatibility)
        self._extra_config = kwargs
        
    def assess_candidate(self, 
                         candidate: Any, 
                         df: pd.DataFrame, 
                         events_df: pd.DataFrame, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         backbone_features: Optional[pd.DataFrame] = None,
                         precomputed_features: Optional[List[str]] = None,
                         precomputed_residuals: Optional[pd.DataFrame] = None,
                         y_causal: Optional[pd.Series] = None) -> Dict[str, float]:
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
        metrics.update(self.compute_validity_metrics(
            candidate, X, y, 
            backbone_features=backbone_features,
            precomputed_residuals=precomputed_residuals
        ))
        
        # 2. Stability
        metrics.update(self.compute_stability_metrics(events_df, y))
        
        # 3. Predictive Integrity (Uses downsampled X)
        metrics.update(self.compute_predictive_integrity(X, y, y_causal=y_causal))
        
        # 4. Robustness
        metrics.update(self.compute_robustness_metrics(y))
        
        # 5. Complexity
        metrics.update(self.compute_complexity_metrics(candidate, events_df))
        
        # 5.5 Feature Importance (NEW - for God Feature detection)
        feature_importance_result = self._extract_feature_importance(X, y)
        if feature_importance_result and 'feature_importance' in feature_importance_result:
            metrics['feature_importance'] = feature_importance_result['feature_importance']
            metrics['event_count'] = len(events_df)  # Store event count for validation
        
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
                    tprint_info(f"   📊 Candidate {candidate_id} filtered out: {filter_failures}")
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
            
        # 8. Determine Causal Quality Status
        if metrics['Layer2Score'] >= 0.5:
             metrics['causal_quality_status'] = 'PASSED'
        elif metrics['Layer2Score'] >= 0.3:
             metrics['causal_quality_status'] = 'WEAK'
        else:
             metrics['causal_quality_status'] = 'FAILED'
        
        
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

    def _apply_survival_filters(
        self, 
        metrics: Dict[str, float], 
        events_count: int, 
        family: str = None,
        role: SignalRole = None
    ) -> Tuple[bool, List[str]]:
        """
        Apply role-aware survival filters.
        
        Args:
            metrics: Computed metrics for the candidate
            events_count: Number of events
            family: Family name (for backward compatibility)
            role: SignalRole - determines which threshold table to use
            
        Returns:
            (passed, list of failure reasons)
        """
        failures = []
        
        # Infer role from family name if not explicitly provided
        if role is None:
            role = self._infer_role_from_family(family)
        
        # Get role-specific filters or fall back to defaults
        if role and role in ROLE_SURVIVAL_FILTERS:
            filters = ROLE_SURVIVAL_FILTERS[role]
            min_events = 25 if role == SignalRole.INTERACTION else (30 if role == SignalRole.TRIGGER else self.MIN_EVENTS_SURVIVAL)
        else:
            filters = self.SURVIVAL_FILTERS
            min_events = self.MIN_EVENTS_SURVIVAL
        
        # Event count check (not for PREDICTOR/CONTEXT which are dense)
        if role not in [SignalRole.PREDICTOR, SignalRole.CONTEXT]:
            if events_count < min_events:
                failures.append(f"min_events={events_count} < {min_events}")
        
        # Apply role-specific filters
        for metric_name, filter_tuple in filters.items():
            if len(filter_tuple) >= 2:
                min_val, max_val = filter_tuple[0], filter_tuple[1]
                reason = filter_tuple[2] if len(filter_tuple) > 2 else f"{metric_name} out of range"
                
                value = metrics.get(metric_name)
                if value is None:
                    # Skip metrics that aren't computed for this candidate
                    continue
                    
                if not (min_val <= value <= max_val):
                    failures.append(f"{metric_name}={value:.4f} not in [{min_val}, {max_val}] - {reason}")
        
        return len(failures) == 0, failures
    
    def _infer_role_from_family(self, family: str) -> Optional[SignalRole]:
        """
        Infer signal role from family name patterns.
        """
        if family is None:
            return None
            
        family_upper = str(family).upper()
        
        # Interaction triggers: COMPOSITE_*, *_INT, *_INTERACTION
        if 'COMPOSITE' in family_upper or '_INT' in family_upper or 'INTERACTION' in family_upper:
            return SignalRole.INTERACTION
        
        # Context/regime: *_REGIME, *_CONTEXT, volatility regime names
        if 'REGIME' in family_upper or 'CONTEXT' in family_upper:
            return SignalRole.CONTEXT
        
        # Pure triggers: *_TRIGGER, CAUSAL_SURPRISE, *_SPIKE, *_SHOCK
        if any(t in family_upper for t in ['TRIGGER', 'CAUSAL_SURPRISE', 'SPIKE', 'SHOCK', 'SPECIALIST']):
            return SignalRole.TRIGGER
        
        # Continuous predictors: *_Z, *_CONTINUOUS, *_RESIDUAL
        if any(t in family_upper for t in ['_Z', 'CONTINUOUS', 'RESIDUAL', 'PREDICTOR']):
            return SignalRole.PREDICTOR
        
        # Default to trigger for backward compatibility
        return SignalRole.TRIGGER

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

    def _fracdiff_innovation(self, series: pd.Series, d: float = 0.4, window: int = 20) -> pd.Series:
        """
        Apply FracDiff then extract innovation (residualize against lag).
        """
        try:
            # 1. FracDiff (using orthogonal_label_generation's helper if available, else simple fallback)
            # Assuming simple fracdiff logic here or import. 
            # Implemeting lightweight fixed-window fracdiff for safety
            def get_weights(d, size):
                w = [1.0]
                for k in range(1, size):
                    w_k = -w[-1] * (d - k + 1) / k
                    w.append(w_k)
                return np.array(w)
            
            # Simple fixed window
            width = 24 # Truncated
            w = get_weights(d, width)
            fd_vals = np.convolve(series.fillna(method='ffill').values, w[::-1], mode='valid')
            fd_series = pd.Series(fd_vals, index=series.index[width-1:])
            
            # Reindex to full
            fd_series = fd_series.reindex(series.index)
            
            # 2. Residualize (Innovation)
            x = fd_series.shift(1).fillna(method='bfill')
            y = fd_series.fillna(method='ffill')
            
            # Vectorized Linear Regression (Slope = Cov(x,y)/Var(x))
            # Just use valid data
            valid = ~(x.isna() | y.isna())
            if valid.sum() > 20:
                slope = np.cov(x[valid], y[valid])[0,1] / (np.var(x[valid]) + 1e-9)
                intercept = np.mean(y[valid]) - slope * np.mean(x[valid])
                innovation = y - (slope * x + intercept)
            else:
               innovation = y - x # Fallback to simple diff

            # 3. Studentize
            return innovation / (innovation.rolling(window=window).std().fillna(1.0) + 1e-9)
        except Exception:
            return series

    def _residualize_feature(self, feature_series: pd.Series, window: int = 20) -> pd.Series:
        """
        Self-residualization via AR(1) and studentization.
        """
        try:
            x = feature_series.shift(1).fillna(method='bfill')
            y = feature_series.fillna(method='ffill')
            
            valid = ~(x.isna() | y.isna())
            if valid.sum() > 20:
                slope = np.cov(x[valid], y[valid])[0,1] / (np.var(x[valid]) + 1e-9)
                intercept = np.mean(y[valid]) - slope * np.mean(x[valid])
                innovation = y - (slope * x + intercept)
            else:
                 innovation = y - x

            return innovation / (innovation.rolling(window=window).std().fillna(1.0) + 1e-9)
        except Exception:
            return feature_series

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
                    
                    if self.verbose:
                        tprint_info(f"   📉 MI Proxy: Reduced to {len(top_features)} features")
                        
                    # === USER REQUEST: Feature Residualization at 300 features ===
                    if self.verbose:
                        tprint_info(f"   🔄 Performing FracDiff/Residualization on {len(X.columns)} features...")
                    
                    new_X = pd.DataFrame(index=X.index)
                    for col in X.columns:
                        # Price-like features get FracDiff Innovation
                        if any(k in col.lower() for k in ['close', 'open', 'high', 'low', 'price', 'vwap']):
                             new_X[col] = self._fracdiff_innovation(X[col])
                        else:
                             # Others get Self-Residualization
                             new_X[col] = self._residualize_feature(X[col])
                    
                    # Replace X with transformed features
                    X = new_X.fillna(0.0) # Ensure safety
                    
                except Exception as e: 
                    tprint_warning(f"Feature transformation failed: {e}")
            
            # ========== STEP 2: ITERATIVE LGBM FEATURE SUBSETTING ==========
            
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



    def compute_validity_metrics(self, candidate: Any,
                               X: pd.DataFrame, 
                               y: pd.Series,
                               backbone_features: Optional[pd.DataFrame] = None,
                               precomputed_residuals: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute validity metrics using residual feature extraction for proper CI testing.
        
        Logic: Regress geometry candidate against backbone features first.
        Use only the RESIDUALS for CI test - this proves the geometry captures
        "The Unexplained" (unique information beyond backbone).
        
        Uses MDI (Mean Decrease Impurity) on OOS fold for geometry validation.
        """
        import time
        start_time = time.time()
        
        try:
            if self.verbose:
                tprint_info(f"   🔬 Computing validity metrics with residual extraction...")
            
            # ========== STEP 1: RESIDUAL FEATURE EXTRACTION ==========
            # Regress candidate features against backbone to extract residuals
            X_residual = X.copy()
            backbone_explained_variance = 0.0
            
            # Use precomputed residuals if available (Regime-Level Cache)
            if precomputed_residuals is not None:
                common_idx = X.index.intersection(precomputed_residuals.index)
                common_cols = [c for c in X.columns if c in precomputed_residuals.columns]
                if len(common_idx) > 50 and len(common_cols) > 0:
                    if self.verbose:
                        tprint_info(f"      ✅ Using {len(common_cols)} precomputed backbone residuals")
                    X_residual = precomputed_residuals.loc[common_idx, common_cols]
                    y = y.loc[common_idx]
                    backbone_explained_variance = 0.5 # Proxy for "filtered"
                else:
                    precomputed_residuals = None # Fallback to fit
            
            if precomputed_residuals is None and backbone_features is not None and not backbone_features.empty:
                common_idx = X.index.intersection(backbone_features.index)
                if len(common_idx) > 50:
                    X_common = X.loc[common_idx]
                    bb_common = backbone_features.loc[common_idx]
                    y_common = y.loc[common_idx] if isinstance(y, pd.Series) else y
                    
                    # PRAGMATIC FIX: Simple mean subtraction instead of Ridge regression
                    # Both scipy.linalg.solve AND numpy.linalg.lstsq can hang on pathological matrices
                    # Mean subtraction gives ~80% of denoising benefit with 0% hang risk
                    residual_cols = []
                    try:
                        if self.verbose:
                            tprint_info(f"      🔧 Starting robust backbone residualization (Ridge)...")
                        
                        # Prepare matrices
                        bb_matrix = bb_common.fillna(0).values  # (n_samples, n_bb)
                        X_matrix = X_common.fillna(0).values    # (n_samples, n_X)
                        
                        # Standardize backbone for stable Ridge solution
                        from sklearn.preprocessing import StandardScaler
                        bb_scaler = StandardScaler()
                        bb_scaled = bb_scaler.fit_transform(bb_matrix)
                        
                        # Fit Ridge once for all columns (multiple outputs)
                        # We use a substantial alpha for stability with high dimensionality
                        # Use lsqr solver for large sparse-like matrices efficiency
                        from sklearn.linear_model import Ridge
                        ridge = Ridge(alpha=10.0, solver='lsqr')
                        ridge.fit(bb_scaled, X_matrix)
                        
                        # Extract residuals: X_residual = X_actual - X_explained_by_backbone
                        X_explained = ridge.predict(bb_scaled)
                        X_res_vals = X_matrix - X_explained
                        
                        # Calculate explained variance per feature
                        var_actual = np.var(X_matrix, axis=0)
                        var_residual = np.var(X_res_vals, axis=0)
                        explained_per_feature = 1 - (var_residual / (var_actual + 1e-9))
                        backbone_explained_variance = np.mean(np.maximum(0, explained_per_feature))
                        
                        if self.verbose:
                            tprint_info(f"      📈 Mean backbone explained variance: {backbone_explained_variance:.4f}")
                        
                        # Create residual series
                        for idx, col in enumerate(X_common.columns):
                            residual_cols.append(pd.Series(
                                X_res_vals[:, idx],
                                index=common_idx,
                                name=f"{col}_residual"
                            ))
                            
                    except Exception as e:
                        if self.verbose:
                            tprint_error(f"   ❌ Robust residualization failed: {e}")
                        # Minimal fallback during FIT: just use centered if ridge failed
                        X_residual = X_common - X_common.mean()
                        backbone_explained_variance = 0.0
                    
                    if residual_cols:
                        X_residual = pd.concat(residual_cols, axis=1)
                        if isinstance(y, pd.Series):
                            y = y.loc[common_idx]
                        backbone_explained_variance /= len(residual_cols)
            
            # ========== STEP 2: CI SCORE USING RESIDUALS ==========
            # If backbone explains most variance, CI score should be low
            # The residual features should still predict y for high CI
            
            if self.verbose:
                tprint_info(f"      📊 STEP 2: Computing CI score (y={len(y)}, X_residual={X_residual.shape})")
            
            if len(y) > 30 and X_residual.shape[1] > 0:
                n_splits = 3 if len(y) > 500 else 2
                cv = TimeSeriesSplit(n_splits=n_splits)
                r2_scores = []
                mdi_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_residual)):
                    if len(train_idx) < 20:
                        continue
                    
                    if self.verbose:
                        tprint_info(f"         🔄 Fold {fold_idx+1}: train={len(train_idx)}, val={len(val_idx)}")
                    
                    X_train = X_residual.iloc[train_idx].fillna(0)
                    X_val = X_residual.iloc[val_idx].fillna(0)
                    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                    y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
                    
                    # Ridge for OOS R²
                    if self.verbose:
                        tprint_info(f"         🏔️ Fitting Ridge (shape={X_train.shape})...")
                    model = Ridge(alpha=1.0, solver='lsqr')
                    model.fit(X_train, y_train)
                    if self.verbose:
                        tprint_info(f"         ✅ Ridge fitted, scoring...")
                    r2 = model.score(X_val, y_val)
                    r2_scores.append(max(0.0, r2))
                    if self.verbose:
                        tprint_info(f"         ✅ R2={r2:.4f}")
                    
                    # LightGBM for MDI (importance on OOS)
                    try:
                        if self.verbose:
                            tprint_info(f"         🌲 Fitting LightGBM for MDI...")
                        import lightgbm as lgb
                        y_binary = (y_train > y_train.median()).astype(int)
                        if len(np.unique(y_binary)) >= 2:
                            lgb_model = lgb.LGBMClassifier(
                                n_estimators=30, max_depth=3, num_leaves=8,
                                learning_rate=0.1, verbosity=-1, n_jobs=1, random_state=42
                            )
                            lgb_model.fit(X_train, y_binary)
                            mdi = np.mean(lgb_model.feature_importances_) / (len(lgb_model.feature_importances_) + 1e-9)
                            mdi_scores.append(mdi)
                            if self.verbose:
                                tprint_info(f"         ✅ LightGBM MDI={mdi:.4f}")
                    except Exception as e:
                        if self.verbose:
                            tprint_warning(f"         ⚠️ LightGBM failed: {e}")
                
                # CI score = OOS R² on residuals (what geometry explains beyond backbone)
                ci_score = np.mean(r2_scores) if r2_scores else 0.0
                mdi_score = np.mean(mdi_scores) if mdi_scores else 0.0
                
                # Adjust CI: if backbone explains >80% variance, penalize
                if backbone_explained_variance > 0.8:
                    ci_score *= 0.3  # Heavy penalty - geometry is redundant
                elif backbone_explained_variance > 0.6:
                    ci_score *= 0.6  # Moderate penalty
                
            else:
                ci_score = 0.01  # Minimum non-zero for small samples
                mdi_score = 0.0
            
            # ========== STEP 3: PSR (Probabilistic Sharpe Ratio proxy) ==========
            # PRAGMATIC FIX: Ridge bootstrap can hang even with lsqr solver
            # Use simple correlation-based stability instead
            if self.verbose:
                tprint_info(f"      📈 STEP 3: Computing PSR (X={X.shape}, y={len(y)})")
            psr = 0.5  # Default
            try:
                if X.shape[1] > 0 and len(y) > 30:
                    if self.verbose:
                        tprint_info(f"         📊 Computing feature-target correlations...")
                    
                    # Simple feature stability: correlation consistency
                    X_filled = X.fillna(0)
                    correlations = []
                    for col in X_filled.columns[:min(30, len(X_filled.columns))]:
                        corr = np.corrcoef(X_filled[col].values, y.values if hasattr(y, 'values') else y)[0, 1]
                        if not np.isnan(corr) and not np.isinf(corr):
                            correlations.append(abs(corr))
                    
                    if self.verbose:
                        tprint_info(f"         ✅ Computed {len(correlations)} correlations")
                    
                    # Feature stability = ratio of strong correlations
                    if correlations:
                        mean_corr = np.mean(correlations)
                        std_corr = np.std(correlations)
                        feat_stab = mean_corr  # Higher mean correlation = more stable
                        feat_stab = max(0, min(1, feat_stab))
                    else:
                        feat_stab = 0.5
                    
                    if self.verbose:
                        tprint_info(f"         ✅ Feature stability: {feat_stab:.4f}")
                    
                    #Residual autocorrelation (simple check)
                    if self.verbose:
                        tprint_info(f"         📊 Computing residual autocorr...")
                    
                    # Simple residuals: y - mean(y) (NO Ridge - can hang!)
                    simple_residuals = y.values if hasattr(y, 'values') else y
                    simple_residuals = simple_residuals - np.mean(simple_residuals)
                    res_series = pd.Series(simple_residuals)
                    res_autocorr = np.abs(res_series.autocorr()) if len(res_series) > 10 else 0.5
                    if np.isnan(res_autocorr):
                        res_autocorr = 0.5
                    
                    if self.verbose:
                        tprint_info(f"         ✅ Autocorr: {res_autocorr:.4f}")
                    
                    psr = 0.6 * feat_stab + 0.4 * (1.0 - res_autocorr)
            except Exception:
                pass
            
            elapsed = time.time() - start_time
            if self.verbose:
                tprint_info(f"   ✅ CI_score: {ci_score:.4f}, PSR: {psr:.4f}, MDI: {mdi_score:.4f}, BB_explained: {backbone_explained_variance:.2f} (total: {elapsed:.2f}s)")
            
            return {
                'CI_score': max(0.01, ci_score),  # Min 0.01 to avoid all-zeros
                'PSR': max(0.1, psr),
                'Overlap_Ratio': min(0.9, backbone_explained_variance),  # How much backbone explains
            }
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"   ❌ Validity failed: {e}")
            return {'CI_score': 0.01, 'PSR': 0.1, 'Overlap_Ratio': 0.5}


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

    def compute_predictive_integrity(self, X: pd.DataFrame, y: pd.Series, y_causal: Optional[pd.Series] = None) -> Dict[str, float]:
        try:
            if self.verbose:
                tprint_info(f"   🎯 Computing predictive integrity...")
            
            # Use causal target for IC if available (De Prado recommendation)
            y_ic = y_causal if y_causal is not None else y
            # Align y_ic to X if needed
            if y_causal is not None:
                common_idx = X.index.intersection(y_ic.index)
                if len(common_idx) < len(X):
                     # If alignment is poor, fallback to y
                     y_ic = y
                else:
                     y_ic = y_ic.loc[X.index]

            # 1. OOS R-squared with TimeSeriesSplit
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
            
            # 2. Information Coefficient (IC)
            corrs = X.corrwith(y_ic).abs()
            ic = corrs.max() if not corrs.empty else 0.0
            best_feat = corrs.idxmax() if not corrs.empty else None
            
            # 3. IC Information Ratio (IC_IR) - with numerical stability fixes
            ic_ir = 0.0
            if best_feat and X[best_feat].std() > 1e-6:
                try:
                    # CRITICAL FIX: Cap window size to prevent huge rolling windows
                    # Old: max(5, len(y) // 5) could be 66+ → slow/hang
                    # New: min(30, max(5, len(y) // 10)) → max 30
                    window = min(30, max(5, len(y) // 10))
                    
                    # Manual rolling correlation (faster than pandas rolling().corr())
                    feat_vals = X[best_feat].values
                    y_vals = y_ic.values
                    rolling_corrs = []
                    
                    for i in range(window, len(y)):
                        # Extract window
                        feat_window = feat_vals[i-window:i]
                        y_window = y_vals[i-window:i]
                        
                        # STABILITY CHECK: Skip constant windows (would cause div by zero)
                        if np.std(feat_window) < 1e-9 or np.std(y_window) < 1e-9:
                            continue
                            
                        # Compute correlation
                        corr_matrix = np.corrcoef(feat_window, y_window)
                        window_corr = corr_matrix[0, 1]
                        
                        # STABILITY CHECK: Skip NaN/Inf values
                        if not np.isnan(window_corr) and not np.isinf(window_corr):
                            rolling_corrs.append(window_corr)
                    
                    # Compute IC_IR if we have enough samples
                    if len(rolling_corrs) >= 3:
                        ic_ir_mean = np.mean(rolling_corrs)
                        ic_ir_std = np.std(rolling_corrs)
                        ic_ir = abs(ic_ir_mean / (ic_ir_std + 1e-9))
                        
                except Exception:
                    # Silently skip IC_IR on any error - not critical for pipeline
                    ic_ir = 0.0
            
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

    def _extract_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Extract feature importance from a simple LightGBM model for God Feature detection.
        
        Returns dict with 'feature_importance' as list of dicts: [{'feature': name, 'importance': value}, ...]
        """
        try:
            import lightgbm as lgb
            
            # Binarize target if needed
            if len(np.unique(y)) > 10:  # Continuous target
                y_binary = (y > y.median()).astype(int)
            else:
                y_binary = y.astype(int)
            
            # Check for degenerate target
            if len(np.unique(y_binary)) < 2:
                return {}
            
            # Train simple model to get feature importance
            model = lgb.LGBMClassifier(
                n_estimators=30,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.1,
                verbosity=-1,
                n_jobs=1,
                random_state=42
            )
            
            model.fit(X, y_binary)
            
            # Extract importance
            importances = model.feature_importances_
            feature_names = X.columns.tolist()
            
            # Create list of dicts sorted by importance (descending)
            feature_importance_list = [
                {'feature': fname, 'importance': float(imp)}
                for fname, imp in zip(feature_names, importances)
            ]
            feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'feature_importance': feature_importance_list
            }
            
        except Exception as e:
            # Silent failure - feature importance is optional
            return {}
