"""
Gamma Specialist: ExtraTrees regression for volatility magnitude prediction.

Predicts realized volatility over the next 6 hours to enable dynamic
risk adjustment (stop-loss, take-profit, position sizing).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from extreme_price_movements.utils import tprint
from extreme_price_movements.fast_funcs import compute_gamma_labels


# Feature set for volatility prediction (25 features)
GAMMA_FEATURE_KEYS = [
    # Current Volatility State
    "atr_pct", "vol_z", "volatility_zscore", "vol_z_30_calm",
    
    # Volatility Dynamics
    "atr_slope", "vol_expansion_ratio", "vol_compression",
    
    # Vol-of-Vol (Regime Change Indicators)
    "vov_ratio", "vov_fast_slow_ratio", "vov_interaction",
    "vov_iqr_20", "vov_mad_20",
    
    # Price Action Intensity
    "range_24h_pct", "range_12h_pct", "rv_24h", "rv_12h", "rv_6h",
    
    # Jump/Shock Indicators
    "jump_rate_10h", "atr_expansion", "accel", "accel_5h",
    
    # Market Context
    "mkt_rv_ratio", "skew",
    
    # Exhaustion Interaction
    "overext", "blowoff_risk",
]


class GammaModel(BaseEstimator, RegressorMixin):
    """
    ExtraTrees Regressor for volatility magnitude prediction.
    """
    
    def __init__(self, n_estimators=300, max_depth=10, n_select=20,
                 min_samples_leaf=50, min_impurity_decrease=1e-5, 
                 ccp_alpha=0.05, random_state=42, n_jobs=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_select = n_select
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.selected_features_ = None
    
    def fit(self, X, y, sample_weight=None):
        """Fit the Gamma model with feature selection."""
        tprint(f"  GammaModel: Running feature selection (target={self.n_select})...")
        
        # 1. Feature Selection (MDI)
        selector = ExtraTreesRegressor(
            n_estimators=50, 
            max_depth=6, 
            max_features="sqrt",
            random_state=self.random_state, 
            n_jobs=self.n_jobs
        )
        selector.fit(X, y, sample_weight=sample_weight)
        
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(self.n_select, X.shape[1])
        self.selected_features_ = X.columns[indices[:top_n]].tolist()
        
        tprint(f"  GammaModel: Selected {len(self.selected_features_)} features")
        
        X_sel = X[self.selected_features_]
        
        # 2. Main Model Training
        tprint(f"  GammaModel: Training ExtraTreesRegressor...")
        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            max_features="sqrt",
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        self.model.fit(X_sel, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X):
        """Predict volatility magnitude."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_sel = X[self.selected_features_]
        return self.model.predict(X_sel)


def compute_gamma_weights(y_gamma, base_weights):
    """
    Compute Huber-style weights to downweight extreme outliers.
    
    Args:
        y_gamma: Target gamma values
        base_weights: Base sample weights
    
    Returns:
        Adjusted weights
    """
    median = np.median(y_gamma)
    mad = np.median(np.abs(y_gamma - median))
    z_score = (y_gamma - median) / (1.4826 * mad + 1e-9)
    
    # Huber threshold: |z| > 2.5 gets downweighted
    huber_weights = np.where(
        np.abs(z_score) > 2.5, 
        2.5 / (np.abs(z_score) + 1e-9), 
        1.0
    )
    
    return base_weights * huber_weights


def train_gamma_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Gamma Specialist regression model.
    
    Args:
        panel: Dictionary with OHLCV DataFrames
        feats: Dictionary of feature DataFrames
        cfg: Configuration dictionary
        syms: List of symbols to train on
        ts_end: End timestamp for training window
    
    Returns:
        Trained GammaModel instance
    """
    tprint("Training Gamma Specialist (ExtraTrees Regression)...")
    
    # 1. Generate gamma labels
    horizon = cfg.get("gamma_horizon", 6)
    tprint(f"  Computing gamma labels (horizon={horizon})...")
    gamma_labels = compute_gamma_labels(panel, feats, horizon=horizon)
    
    # 2. Build feature matrix
    tprint(f"  Building feature matrix ({len(GAMMA_FEATURE_KEYS)} features)...")
    X_list = []
    y_list = []
    
    for sym in syms:
        if sym not in gamma_labels.columns:
            continue
        
        # Get gamma labels for this symbol
        y_sym = gamma_labels[sym].dropna()
        
        if len(y_sym) < 100:
            continue
        
        # Extract features for this symbol
        X_sym_list = []
        valid_idx = []
        
        for idx in y_sym.index:
            if idx not in feats[GAMMA_FEATURE_KEYS[0]].index:
                continue
            
            row = []
            valid = True
            for feat_key in GAMMA_FEATURE_KEYS:
                if feat_key not in feats or sym not in feats[feat_key].columns:
                    valid = False
                    break
                val = feats[feat_key].loc[idx, sym]
                if np.isnan(val) or np.isinf(val):
                    valid = False
                    break
                row.append(val)
            
            if valid:
                X_sym_list.append(row)
                valid_idx.append(idx)
        
        if len(X_sym_list) > 0:
            X_sym = np.array(X_sym_list, dtype=np.float32)
            y_sym_aligned = y_sym.loc[valid_idx].values
            
            X_list.append(X_sym)
            y_list.append(y_sym_aligned)
    
    if not X_list:
        tprint("  ERROR: No valid training data for Gamma Specialist")
        return None
    
    X = pd.DataFrame(np.vstack(X_list), columns=GAMMA_FEATURE_KEYS)
    y = np.concatenate(y_list)
    
    tprint(f"  Training data: {len(X)} samples, {X.shape[1]} features")
    tprint(f"  Gamma range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
    
    # 3. Compute sample weights (Huber-style for robustness)
    base_weights = np.ones(len(y), dtype=np.float32)
    sample_weights = compute_gamma_weights(y, base_weights)
    
    tprint(f"  Applied Huber weighting (downweighted {(sample_weights < 1.0).sum()} outliers)")
    
    # 4. Train model
    model = GammaModel(
        n_estimators=cfg.get("gamma_n_estimators", 300),
        max_depth=cfg.get("gamma_max_depth", 10),
        n_select=cfg.get("gamma_n_select", 20),
        min_samples_leaf=cfg.get("gamma_min_samples_leaf", 50),
        min_impurity_decrease=cfg.get("gamma_min_impurity_decrease", 1e-5),
        ccp_alpha=cfg.get("gamma_ccp_alpha", 0.05),
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", 3)
    )
    
    model.fit(X, y, sample_weight=sample_weights)
    
    # 5. Validation metrics
    y_pred = model.predict(X)
    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    tprint(f"  R² Score: {r2:.3f}")
    tprint(f"  MAE: {mae:.3f}")
    tprint(f"  RMSE: {rmse:.3f}")
    
    # Regime accuracy
    def classify_regime(gamma_val):
        if gamma_val < 0.5:
            return 0  # Dead
        elif gamma_val < 1.0:
            return 1  # Normal
        elif gamma_val < 2.0:
            return 2  # High_Vol
        else:
            return 3  # Explosive
    
    y_regime = np.array([classify_regime(v) for v in y])
    y_pred_regime = np.array([classify_regime(v) for v in y_pred])
    
    regime_acc = (y_regime == y_pred_regime).mean()
    tprint(f"  Regime Classification Accuracy: {regime_acc:.1%}")
    
    tprint("✅ Gamma Specialist training complete")
    
    return model
