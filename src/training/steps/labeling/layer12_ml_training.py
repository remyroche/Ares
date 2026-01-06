"""
Layer-12 ML Training Pipeline
==============================

Extends Layer-12 with ML model training capabilities:
1. Feature Selection (Variance + Correlation filtering)
2. PCA per Component
3. Component-based Base Model Race (5-model)
4. Meta-Learner Input Preparation
5. Comprehensive Metric Tracking

Metrics Tracked:
- Predictive: PR-AUC, IC, OOS_R²
- Stability: IC_IR, CV_freq, Dir_consistency, DSR, SPA_p
- Complexity: Sparsity, Feature_Overlap_Ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
except ImportError:
    def tprint_info(msg): print(f"ℹ️ {msg}")
    def tprint_warning(msg): print(f"⚠️ {msg}")
    def tprint_success(msg): print(f"✅ {msg}")
    def tprint_error(msg): print(f"❌ {msg}")

# Import ML libraries
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# =============================================================================
# METRIC THRESHOLDS
# =============================================================================

METRIC_THRESHOLDS = {
    # Predictive
    'PR_AUC': {'min': 0.1, 'good': 0.3, 'excellent': 0.5},
    'IC': {'min': 0.05, 'good': 0.15, 'excellent': 0.25},
    'OOS_R2': {'min': 0.0, 'good': 0.03, 'excellent': 0.1},
    
    # Stability
    'IC_IR': {'min': 0.5, 'good': 1.0, 'excellent': 2.0},
    'CV_freq': {'max': 0.5, 'good': 0.3, 'excellent': 0.2},
    'Dir_consistency': {'min': 0.6, 'good': 0.8, 'excellent': 0.9},
    'DSR': {'min': 0.5, 'good': 0.8, 'excellent': 1.0},
    'SPA_p': {'max': 0.1, 'good': 0.05, 'excellent': 0.01},
    
    # Complexity
    'Sparsity': {'max': 0.9, 'good': 0.7, 'warning': 0.95},
    'Feature_Overlap_Ratio': {'max': 0.5, 'good': 0.3, 'excellent': 0.1}
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ComponentModelResult:
    """Result of model training for a single component."""
    component_name: str
    model_name: str
    model: Any
    pca_model: Optional[PCA]
    selected_features: List[str]
    metrics: Dict[str, float]
    oof_predictions: pd.Series
    

@dataclass
class Layer12MLOutput:
    """Complete ML training output."""
    component_results: Dict[str, ComponentModelResult]
    meta_X: pd.DataFrame  # Stacked OOF predictions for meta-learner
    metrics_report: pd.DataFrame
    sample_weights: pd.Series
    
    def get_meta_learner_input(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get input for meta-learner training."""
        return self.meta_X, self.sample_weights


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features(
    X: pd.DataFrame,
    min_var: float = 1e-4,
    max_corr: float = 0.95,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature selection workflow:
    1. Remove near-constant features (VarianceThreshold)
    2. Remove highly correlated features
    
    Returns:
        X_selected: Filtered feature matrix
        selected_columns: List of selected column names
    """
    original_cols = list(X.columns)
    
    # 1. Remove near-constant features
    try:
        selector = VarianceThreshold(threshold=min_var)
        X_var = pd.DataFrame(
            selector.fit_transform(X),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
        if verbose:
            tprint_info(f"   ⚡ Variance filter: {len(original_cols)} → {len(X_var.columns)}")
    except Exception as e:
        if verbose:
            tprint_warning(f"   ⚠️ Variance filter failed: {e}")
        X_var = X
    
    # 2. Remove highly correlated features
    try:
        if len(X_var.columns) > 1:
            corr_matrix = X_var.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > max_corr)]
            X_sel = X_var.drop(columns=to_drop)
            if verbose and to_drop:
                tprint_info(f"   ⚡ Correlation filter: {len(X_var.columns)} → {len(X_sel.columns)}")
        else:
            X_sel = X_var
    except Exception as e:
        if verbose:
            tprint_warning(f"   ⚠️ Correlation filter failed: {e}")
        X_sel = X_var
    
    return X_sel, list(X_sel.columns)


def apply_pca(
    X: pd.DataFrame,
    n_components: float = 0.95,
    verbose: bool = True
) -> Tuple[pd.DataFrame, PCA]:
    """
    Apply PCA to reduce dimensionality while preserving variance.
    
    Args:
        X: Feature matrix
        n_components: Variance ratio to preserve (0-1) or number of components
        
    Returns:
        X_pca: PCA-transformed features
        pca: Fitted PCA model
    """
    try:
        pca = PCA(n_components=n_components, svd_solver='full')
        X_pca = pd.DataFrame(
            pca.fit_transform(X),
            index=X.index,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        if verbose:
            tprint_info(f"   📊 PCA: {X.shape[1]} → {pca.n_components_} components ({pca.explained_variance_ratio_.sum():.1%} variance)")
        return X_pca, pca
    except Exception as e:
        if verbose:
            tprint_warning(f"   ⚠️ PCA failed: {e}")
        # Return identity transform
        return X, None


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_comprehensive_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: Optional[pd.Series] = None,
    sample_weight: Optional[pd.Series] = None,
    n_folds: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for model evaluation.
    
    Metrics:
    - Predictive: PR_AUC, IC, OOS_R2
    - Stability: IC_IR, CV_freq, Dir_consistency, DSR
    - Complexity: Sparsity
    """
    metrics = {}
    
    # Align data
    common_idx = y_true.dropna().index.intersection(y_pred.dropna().index)
    y_true = y_true.loc[common_idx]
    y_pred = y_pred.loc[common_idx]
    
    if len(y_true) < 50:
        return {'error': 'Insufficient samples'}
    
    # ========== PREDICTIVE METRICS ==========
    
    # PR-AUC (if probabilities available)
    if y_prob is not None:
        y_prob = y_prob.loc[common_idx]
        try:
            metrics['PR_AUC'] = average_precision_score(y_true, y_prob)
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['PR_AUC'] = 0.0
            metrics['ROC_AUC'] = 0.5
    
    # IC (Spearman correlation)
    try:
        ic, _ = spearmanr(y_pred, y_true)
        metrics['IC'] = ic if not np.isnan(ic) else 0.0
    except Exception:
        metrics['IC'] = 0.0
    
    # ========== STABILITY METRICS ==========
    
    # IC_IR (IC stability over rolling windows)
    try:
        window = max(50, len(y_true) // 10)
        rolling_ic = pd.Series(index=y_true.index, dtype=float)
        for i in range(window, len(y_true)):
            window_true = y_true.iloc[i-window:i]
            window_pred = y_pred.iloc[i-window:i]
            ic_val, _ = spearmanr(window_pred, window_true)
            rolling_ic.iloc[i] = ic_val
        
        ic_mean = rolling_ic.dropna().mean()
        ic_std = rolling_ic.dropna().std()
        metrics['IC_IR'] = ic_mean / (ic_std + 1e-9) if ic_std > 0 else 0.0
    except Exception:
        metrics['IC_IR'] = 0.0
    
    # Dir_consistency (directional agreement)
    try:
        pred_sign = np.sign(y_pred)
        true_sign = np.sign(y_true)
        metrics['Dir_consistency'] = (pred_sign == true_sign).mean()
    except Exception:
        metrics['Dir_consistency'] = 0.5
    
    # DSR (Deflated Sharpe Ratio)
    try:
        pseudo_returns = y_pred * y_true  # Profit when aligned
        sharpe = pseudo_returns.mean() / (pseudo_returns.std() + 1e-9)
        n_trials = 100  # Assume 100 backtests
        metrics['DSR'] = sharpe * np.sqrt(1 - 1/n_trials)  # Simple deflation
    except Exception:
        metrics['DSR'] = 0.0
    
    # ========== COMPLEXITY METRICS ==========
    
    # Sparsity
    try:
        near_zero = (y_pred.abs() < 1e-6).sum()
        metrics['Sparsity'] = near_zero / len(y_pred)
    except Exception:
        metrics['Sparsity'] = 0.0
    
    return metrics


def compute_cv_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_splits: int = 3,
    sample_weight: Optional[pd.Series] = None
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Compute cross-validated metrics.
    
    Returns:
        metrics: Aggregated CV metrics
        oof_predictions: Out-of-fold predictions
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = pd.Series(np.nan, index=y.index)
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Get sample weights if available
            sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None
            
            # Fit model
            if sw_train is not None:
                model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.predict(X_val)
            
            oof_pred.iloc[val_idx] = y_prob
            
            # Compute fold metrics
            fold_m = compute_comprehensive_metrics(y_val, pd.Series(y_prob, index=y_val.index), pd.Series(y_prob, index=y_val.index))
            fold_metrics.append(fold_m)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Fold {fold_idx} failed: {e}")
            continue
    
    # Aggregate metrics
    if fold_metrics:
        agg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [fm.get(key, 0) for fm in fold_metrics if key in fm]
            if values:
                agg_metrics[key] = np.mean(values)
                agg_metrics[f"{key}_std"] = np.std(values)
    else:
        agg_metrics = {}
    
    return agg_metrics, oof_pred


# =============================================================================
# MODEL RACE
# =============================================================================

def create_base_models() -> Dict[str, Any]:
    """
    Create base models for the 5-model race.
    Uses same models as label_based_layer_2.py.
    """
    models = {}
    
    if LGBM_AVAILABLE:
        models['LGBM_Focal'] = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1
        )
        models['LGBM_BCE'] = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbose=-1
        )
    
    if XGB_AVAILABLE:
        models['XGB_Tree'] = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='aucpr',
            verbosity=0
        )
        models['XGB_Linear'] = xgb.XGBClassifier(
            objective='binary:logistic',
            booster='gblinear',
            n_estimators=100,
            learning_rate=0.1,
            eval_metric='aucpr',
            verbosity=0
        )
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            verbose=0
        )
    
    return models


def compute_tier_coverage(
    y_pred: pd.Series,
    tier_weights: pd.Series,
    threshold: float = 0.5
) -> float:
    """
    Compute Tier-1 coverage: fraction of Tier-1 events correctly identified.
    Penalizes models that ignore rare extreme events.
    
    Args:
        y_pred: Model predictions (probability or binary)
        tier_weights: Tier weights (1.0 for Tier-1, 0.5 for Tier-2)
        threshold: Classification threshold
        
    Returns:
        Tier-1 coverage ratio (0-1)
    """
    # Identify Tier-1 events (weight >= 0.9)
    tier1_mask = tier_weights >= 0.9
    tier1_count = tier1_mask.sum()
    
    if tier1_count == 0:
        return 1.0  # No Tier-1 events, full coverage by default
    
    # Check how many Tier-1 events are captured (predicted positive)
    y_pred_binary = (y_pred >= threshold).astype(float)
    tier1_captured = (y_pred_binary[tier1_mask] == 1).sum()
    
    return tier1_captured / tier1_count


def normalize_metric(value: float, min_val: float, max_val: float, inverse: bool = False) -> float:
    """
    Normalize a metric to [0, 1] range.
    
    Args:
        value: Raw metric value
        min_val: Minimum expected value
        max_val: Maximum expected value
        inverse: If True, lower values are better (e.g., SPA_p)
        
    Returns:
        Normalized value in [0, 1]
    """
    if max_val <= min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    
    if inverse:
        normalized = 1 - normalized
    
    return normalized


def run_model_race(
    X: pd.DataFrame,
    y: pd.Series,
    tier_weights: Optional[pd.Series] = None,
    sample_weight: Optional[pd.Series] = None,
    n_cv_splits: int = 3,
    verbose: bool = True
) -> Tuple[str, Any, Dict[str, float], pd.Series]:
    """
    Run 5-model race and select best using normalized composite score.
    
    NO RFE/feature selection in the race - just model training and evaluation.
    
    Score Formula (all normalized to [0,1]):
        0.35 × PR_AUC + 0.25 × IC_IR + 0.20 × Dir_consistency + 
        0.15 × Tier_coverage + 0.05 × OOS_R²
    
    Returns:
        best_model_name: Name of winning model
        best_model: Fitted model
        best_metrics: All metrics of winning model
        oof_predictions: OOF predictions from winning model
    """
    models = create_base_models()
    
    if not models:
        raise ValueError("No ML models available")
    
    results = {}
    
    for model_name, model in models.items():
        try:
            if verbose:
                tprint_info(f"   🏃 Racing {model_name}...")
            
            # Compute CV metrics (no feature selection, just raw training)
            metrics, oof_pred = compute_cv_metrics(X, y, model, n_cv_splits, sample_weight)
            
            # Add Tier-coverage if tier_weights provided
            if tier_weights is not None:
                tier_cov = compute_tier_coverage(oof_pred, tier_weights)
                metrics['Tier_coverage'] = tier_cov
            else:
                metrics['Tier_coverage'] = 0.5  # Neutral if no tier weights
            
            # Compute OOS R² (regression predictive power)
            try:
                # Simple OOS R² from predictions
                ss_res = ((y - oof_pred.fillna(y.mean())) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                oos_r2 = max(0, 1 - ss_res / (ss_tot + 1e-9))
                metrics['OOS_R2'] = oos_r2
            except Exception:
                metrics['OOS_R2'] = 0.0
            
            # ========== NORMALIZED SCORING ==========
            # All metrics normalized to [0, 1] before combining
            
            pr_auc_norm = normalize_metric(metrics.get('PR_AUC', 0), 0.0, 1.0)
            ic_ir_norm = normalize_metric(metrics.get('IC_IR', 0), 0.0, 3.0)  # Cap at 3.0
            dir_norm = normalize_metric(metrics.get('Dir_consistency', 0), 0.5, 1.0)
            tier_cov_norm = normalize_metric(metrics.get('Tier_coverage', 0), 0.0, 1.0)
            oos_r2_norm = normalize_metric(metrics.get('OOS_R2', 0), 0.0, 0.2)  # Cap at 0.2
            
            # Composite score with weights
            score = (
                0.35 * pr_auc_norm +
                0.25 * ic_ir_norm +
                0.20 * dir_norm +
                0.15 * tier_cov_norm +
                0.05 * oos_r2_norm
            )
            
            # Store normalized values too
            metrics['PR_AUC_norm'] = pr_auc_norm
            metrics['IC_IR_norm'] = ic_ir_norm
            metrics['Dir_consistency_norm'] = dir_norm
            metrics['Tier_coverage_norm'] = tier_cov_norm
            metrics['OOS_R2_norm'] = oos_r2_norm
            metrics['composite_score'] = score
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'oof_pred': oof_pred,
                'score': score
            }
            
            if verbose:
                tprint_info(f"      PR-AUC={metrics.get('PR_AUC', 0):.3f}({pr_auc_norm:.2f}), "
                           f"IC_IR={metrics.get('IC_IR', 0):.2f}({ic_ir_norm:.2f}), "
                           f"Tier={metrics.get('Tier_coverage', 0):.2f}, Score={score:.4f}")
                
        except Exception as e:
            if verbose:
                tprint_warning(f"   ⚠️ {model_name} failed: {e}")
            continue
    
    if not results:
        raise ValueError("All models failed")
    
    # Select best
    best_name = max(results, key=lambda k: results[k]['score'])
    best = results[best_name]
    
    if verbose:
        tprint_success(f"   🏆 Winner: {best_name} (Score={best['score']:.4f})")
    
    return best_name, best['model'], best['metrics'], best['oof_pred']



# =============================================================================
# COMPONENT-BASED TRAINING
# =============================================================================

def group_features_by_component(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group features by component/family prefix.
    """
    component_map = {}
    
    prefixes = ['META_REINFORCED', 'META_WEIGHTED_SUM', 'META_EWMA_1D', 'META_SUM_4H', 'META_MAX_4H',
                'COMPOSITE', 'CAUSAL_SURPRISE', 'SPECIALIST', 'REGIME']
    
    for prefix in prefixes:
        cols = [c for c in X.columns if prefix in c]
        if cols:
            component_map[prefix] = cols
    
    # Catch remaining
    assigned = set(c for cols in component_map.values() for c in cols)
    remaining = [c for c in X.columns if c not in assigned]
    if remaining:
        component_map['OTHER'] = remaining
    
    return component_map


def train_component_models(
    X: pd.DataFrame,
    y: pd.Series,
    tier_weights: Optional[pd.Series] = None,
    sample_weight: Optional[pd.Series] = None,
    use_pca: bool = True,
    pca_variance: float = 0.95,
    verbose: bool = True
) -> Dict[str, ComponentModelResult]:
    """
    Train models per component with PCA (optional).
    
    NO RFE during model race - only pre-race feature selection.
    
    Returns:
        Dictionary of component -> ComponentModelResult
    """
    component_map = group_features_by_component(X)
    results = {}
    
    # Identify backbone features (Specialists and Regimes)
    backbone_features = []
    if 'SPECIALIST' in component_map:
        backbone_features.extend(component_map['SPECIALIST'])
    if 'REGIME' in component_map:
        backbone_features.extend(component_map['REGIME'])
    
    # Also catch PC1, PC2, PC3 explicitly if not assigned to SPECIALIST
    pc_features = [c for c in X.columns if any(p in c for p in ['_PC1', '_PC2', '_PC3', 'rv_z_short'])]
    backbone_features = list(set(backbone_features) | set(pc_features))
    
    if verbose:
        tprint_info(f"🔧 Training models for {len(component_map)} components...")
        tprint_info(f"   🏛️ Structural Backbone: {len(backbone_features)} features")
    
    for comp_name, features in component_map.items():
        # Skip training separate models for the backbone itself (they are used as predictors)
        if comp_name in ['SPECIALIST', 'REGIME']:
            continue
            
        try:
            if verbose:
                tprint_info(f"\n📦 Component: {comp_name} ({len(features)} features)")
            
            # Combine family features with backbone context (Two-Channel Model)
            X_family = X[features]
            X_back = X[backbone_features]
            
            # --- TWO-CHANNEL INTERACTION GENERATION ---
            # Explicitly create Signal x Context features to help tree models
            # We condition every geometry signal on the primary market drivers
            
            # 1. Identify Context Drivers (PC1 of Volatility & Liquidity)
            # Look for 'VOL' and 'PC1', or 'LIQ' and 'PC1' in backbone
            vol_drivers = [c for c in backbone_features if 'VOL' in c and 'PC1' in c]
            liq_drivers = [c for c in backbone_features if 'LIQ' in c and 'PC1' in c]
            
            # Fallback to any PC1 if specific ones not found
            if not vol_drivers and not liq_drivers:
                context_drivers = [c for c in backbone_features if '_PC1' in c][:2]
            else:
                context_drivers = (vol_drivers[:1] + liq_drivers[:1])
            
            X_interact = pd.DataFrame(index=X.index)
            
            if context_drivers:
                if verbose:
                    tprint_info(f"   🧬 Generatiing interactions with Context Drivers: {context_drivers}")
                
                # Vectorized interaction generation
                # For each driver, multiply all family features
                for driver in context_drivers:
                    # Use numpy broadcasting for speed
                    driver_val = X_back[driver].values[:, None] # (N, 1)
                    family_vals = X_family.values               # (N, F)
                    
                    interact_vals = family_vals * driver_val
                    
                    # Create column names
                    interact_cols = [f"{col}_x_{driver}" for col in X_family.columns]
                    
                    # Add to DataFrame
                    chunk_df = pd.DataFrame(interact_vals, index=X.index, columns=interact_cols)
                    X_interact = pd.concat([X_interact, chunk_df], axis=1)

            # Concatenate all channels: Signal + Context + Interactions
            X_comp = pd.concat([X_family, X_back, X_interact], axis=1)
            
            # Remove duplicates if any
            X_comp = X_comp.loc[:, ~X_comp.columns.duplicated()]
            
            # Pre-race feature selection (basic variance + correlation filter)
            X_sel, selected_features = select_features(X_comp, verbose=verbose)
            
            if len(X_sel.columns) == 0:
                if verbose:
                    tprint_warning(f"   ⚠️ No features remaining after selection")
                continue
            
            # PCA (optional)
            pca_model = None
            if use_pca and len(X_sel.columns) > 5:
                X_train, pca_model = apply_pca(X_sel, n_components=pca_variance, verbose=verbose)
            else:
                X_train = X_sel
            
            # Run model race (NO RFE here - just model training)
            best_name, best_model, metrics, oof_pred = run_model_race(
                X_train, y, 
                tier_weights=tier_weights,
                sample_weight=sample_weight, 
                verbose=verbose
            )
            
            results[comp_name] = ComponentModelResult(
                component_name=comp_name,
                model_name=best_name,
                model=best_model,
                pca_model=pca_model,
                selected_features=selected_features,
                metrics=metrics,
                oof_predictions=oof_pred
            )
            
        except Exception as e:
            if verbose:
                tprint_error(f"   ❌ Component {comp_name} failed: {e}")
            continue
    
    return results



# =============================================================================
# META-LEARNER PREPARATION
# =============================================================================

def prepare_meta_learner_input(
    component_results: Dict[str, ComponentModelResult],
    X_structural: Optional[pd.DataFrame] = None,
    regime_features: Optional[pd.DataFrame] = None,
    chaser_output: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Prepare full meta-learner input with all required features.
    
    Meta-Learner Inputs:
    1. All base model predictions (OOF probabilities/scores)
    2. High-IC parent signals (Tier-1 signals from Layer-12)
    3. Regime info (volatility, liquidity, trend)
    4. Agreement features (model disagreement signals)
    5. Chaser outputs
    
    Args:
        component_results: Results from component training
        X_structural: Optional high-IC structural parents
        regime_features: Optional regime DataFrame (R from Layer-12)
        chaser_output: Optional chaser predictions
        
    Returns:
        meta_X: Complete meta-learner feature DataFrame
    """
    meta_features = {}
    
    # 1. Base model OOF predictions
    for comp_name, result in component_results.items():
        meta_features[f"OOF_{comp_name}"] = result.oof_predictions
    
    # Create DataFrame
    meta_X = pd.DataFrame(meta_features)
    
    # 2. Agreement features (model disagreement)
    if len(component_results) > 1:
        oof_cols = [f"OOF_{c}" for c in component_results.keys()]
        oof_df = meta_X[oof_cols]
        
        # Mean prediction across models
        meta_X['OOF_mean'] = oof_df.mean(axis=1)
        
        # Std of predictions (disagreement signal)
        meta_X['OOF_std'] = oof_df.std(axis=1)
        
        # Agreement ratio: models predicting same direction
        positive_votes = (oof_df > 0.5).sum(axis=1)
        meta_X['OOF_agreement_ratio'] = positive_votes / len(oof_cols)
        
        # Max confidence (highest prediction)
        meta_X['OOF_max_conf'] = oof_df.max(axis=1)
        
        # Min confidence (most uncertain model)
        meta_X['OOF_min_conf'] = oof_df.min(axis=1)
    
    # 3. Add structural parents if provided
    if X_structural is not None:
        meta_X = pd.concat([meta_X, X_structural], axis=1)
    
    # 4. Add regime features if provided
    if regime_features is not None:
        meta_X = pd.concat([meta_X, regime_features], axis=1)
    
    # 5. Add chaser output if provided
    if chaser_output is not None:
        meta_X['CHASER_OUTPUT'] = chaser_output
    
    return meta_X.fillna(0)


def prepare_chaser_features(
    tier_weights: pd.DataFrame,
    component_results: Dict[str, ComponentModelResult],
    X_structural: Optional[pd.DataFrame] = None,
    max_features: int = 50
) -> pd.DataFrame:
    """
    Prepare Chaser (Layer 2.5) features.
    
    Chaser Inputs:
    1. Tier-weights (Tier-1/Tier-2) from Layer-12
    2. Composite features (aggregated from base models)
    3. Momentum / Volatility / Volume / Liquidity (passed via X_structural)
    
    Args:
        tier_weights: W matrix from Layer-12
        component_results: Base model results
        X_structural: Additional features (momentum, vol, etc.)
        max_features: Max features for LGBM MDI selection
        
    Returns:
        chaser_X: Chaser feature DataFrame
    """
    chaser_features = {}
    
    # 1. Tier-weight features (mean per sample)
    if tier_weights is not None and len(tier_weights.columns) > 0:
        chaser_features['TIER_mean'] = tier_weights.mean(axis=1)
        chaser_features['TIER_max'] = tier_weights.max(axis=1)
        chaser_features['TIER_count_high'] = (tier_weights >= 0.9).sum(axis=1)
        chaser_features['TIER_count_med'] = ((tier_weights >= 0.4) & (tier_weights < 0.9)).sum(axis=1)
    
    # 2. Aggregated base model outputs
    for comp_name, result in component_results.items():
        # Direct OOF
        chaser_features[f"BASE_{comp_name}"] = result.oof_predictions
    
    # Aggregate stats
    if component_results:
        oof_df = pd.DataFrame({f"BASE_{c}": r.oof_predictions for c, r in component_results.items()})
        chaser_features['BASE_mean'] = oof_df.mean(axis=1)
        chaser_features['BASE_std'] = oof_df.std(axis=1)
        chaser_features['BASE_max'] = oof_df.max(axis=1)
    
    chaser_X = pd.DataFrame(chaser_features)
    
    # 3. Add structural features if provided
    if X_structural is not None:
        chaser_X = pd.concat([chaser_X, X_structural], axis=1)
    
    # Count features
    n_features = len(chaser_X.columns)
    
    # Note: If n_features > max_features, caller should apply LGBM MDI selection
    # This is flagged but not implemented here to avoid circular dependency
    
    return chaser_X.fillna(0), n_features



def compute_sample_weights(
    tier_weights: pd.Series,
    magnitude_weights: Optional[pd.Series] = None,
    uniqueness_weights: Optional[pd.Series] = None
) -> pd.Series:
    """
    Compute combined sample weights.
    
    sample_weight = Tier + Magnitude + Uniqueness (normalized)
    """
    weights = tier_weights.copy()
    
    if magnitude_weights is not None:
        weights = weights + magnitude_weights
    
    if uniqueness_weights is not None:
        weights = weights + uniqueness_weights
    
    # Normalize to [0.1, 1.0]
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    weights = 0.1 + 0.9 * weights
    
    return weights


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

class Layer12MLTrainer:
    """
    Complete Layer-12 ML Training Pipeline.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.component_results = {}
        self.metrics_report = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        W: Optional[pd.DataFrame] = None,
        sample_weight: Optional[pd.Series] = None,
        use_pca: bool = True,
        pca_variance: float = 0.95
    ) -> Layer12MLOutput:
        """
        Run full training pipeline.
        
        Args:
            X: Feature matrix from Layer-12
            y: Target labels
            W: Tier-weight matrix (optional)
            sample_weight: Pre-computed sample weights
            use_pca: Apply PCA per component
            pca_variance: Variance to preserve in PCA
            
        Returns:
            Layer12MLOutput with trained models and meta-learner input
        """
        if self.verbose:
            tprint_info("=" * 60)
            tprint_info("🚀 LAYER-12 ML TRAINING PIPELINE")
            tprint_info("=" * 60)
            tprint_info(f"   Features: {X.shape[1]}, Samples: {len(X)}")
        
        # Compute tier_weights mean per sample (for Tier-coverage in model race)
        tier_weights_series = None
        if W is not None:
            tier_weights_series = W.mean(axis=1)
        
        # Compute sample weights if not provided
        if sample_weight is None and W is not None:
            sample_weight = tier_weights_series
            if self.verbose:
                tprint_info(f"   📊 Using tier-weights as sample weights")
        
        # Train component models (with tier_weights for Tier-coverage metric)
        self.component_results = train_component_models(
            X, y, 
            tier_weights=tier_weights_series,
            sample_weight=sample_weight, 
            use_pca=use_pca, 
            pca_variance=pca_variance, 
            verbose=self.verbose
        )

        
        if not self.component_results:
            raise ValueError("No components trained successfully")
        
        # Prepare meta-learner input
        meta_X = prepare_meta_learner_input(self.component_results)
        
        # Build metrics report
        self.metrics_report = self._build_metrics_report()
        
        if self.verbose:
            tprint_success("\n" + "=" * 60)
            tprint_success("✅ LAYER-12 ML TRAINING COMPLETE")
            tprint_success("=" * 60)
            tprint_info(f"   Components trained: {len(self.component_results)}")
            tprint_info(f"   Meta-learner features: {meta_X.shape[1]}")
        
        return Layer12MLOutput(
            component_results=self.component_results,
            meta_X=meta_X,
            metrics_report=self.metrics_report,
            sample_weights=sample_weight if sample_weight is not None else pd.Series(1.0, index=y.index)
        )
    
    def _build_metrics_report(self) -> pd.DataFrame:
        """Build comprehensive metrics report."""
        rows = []
        
        for comp_name, result in self.component_results.items():
            row = {
                'Component': comp_name,
                'Model': result.model_name,
                'Features': len(result.selected_features),
                **result.metrics
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add quality grades
        def grade(val, metric):
            thresholds = METRIC_THRESHOLDS.get(metric, {})
            if 'excellent' in thresholds and val >= thresholds['excellent']:
                return 'A'
            if 'good' in thresholds and val >= thresholds['good']:
                return 'B'
            if 'min' in thresholds and val >= thresholds['min']:
                return 'C'
            return 'D'
        
        for metric in ['PR_AUC', 'IC', 'IC_IR', 'Dir_consistency', 'DSR']:
            if metric in df.columns:
                df[f'{metric}_grade'] = df[metric].apply(lambda x: grade(x, metric))
        
        return df
    
    def get_report_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Layer-12 ML Training Report",
            "",
            "## Component Summary",
            ""
        ]
        
        if self.metrics_report is not None:
            lines.append(self.metrics_report.to_markdown())
        
        lines.extend([
            "",
            "## Metric Thresholds",
            "| Metric | Min | Good | Excellent |",
            "|--------|-----|------|-----------|"
        ])
        
        for metric, thresholds in METRIC_THRESHOLDS.items():
            if 'min' in thresholds:
                lines.append(f"| {metric} | {thresholds.get('min', '-')} | {thresholds.get('good', '-')} | {thresholds.get('excellent', '-')} |")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train_layer12_ml_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    W: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Layer12MLOutput:
    """
    Convenience function to run Layer-12 ML training.
    """
    trainer = Layer12MLTrainer(verbose=verbose)
    return trainer.train(X, y, W)
