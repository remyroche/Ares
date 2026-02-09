"""
Gamma Specialist: ExtraTrees regression for volatility magnitude prediction.

Predicts realized volatility over the next 6 hours to enable dynamic
risk adjustment (stop-loss, take-profit, position sizing).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from extreme_price_movements.utils import tprint
from extreme_price_movements.fast_funcs import compute_gamma_labels
from extreme_price_movements.purged_cv import PurgedKFold


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

    def compute_oof_predictions(self, X, y):
        """Compute OOF predictions for Gamma Model."""
        tprint("  GammaModel: Computing OOF predictions...")

        # Use simple KFold for regression if time-dependency is loose,
        # but PurgedKFold is better for time-series.
        # Using PurgedKFold(n_splits=5)
        kf = PurgedKFold(n_splits=5, purge=2, embargo=0) # minimal purge for speed

        oof_preds = np.full(len(y), np.nan, dtype=np.float32)

        # Ensure array
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
            cols = X.columns
        else:
            X_arr = X
            cols = None

        y_arr = np.array(y, dtype=np.float32)

        # If selected features already known, use them. Else use all?
        # fit() selects features. If we haven't fit, we don't know features.
        # Assume full fit happens later or we do feature selection inside fold?
        # Doing FS inside fold is expensive.
        # We will use all features for OOF if not selected, or pre-select?
        # Let's perform a quick pre-selection on full data first if self.selected_features_ is None.

        if self.selected_features_ is None and cols is not None:
             # Quick fit to get features
             self.fit(X, y) # This sets self.selected_features_

        if self.selected_features_ is not None and cols is not None:
             # Map selected features to indices
             col_idx = [cols.get_loc(c) for c in self.selected_features_]
             X_use = X_arr[:, col_idx]
        else:
             X_use = X_arr

        for i, (train_idx, test_idx) in enumerate(kf.split(X_use)):
            X_train, X_test = X_use[train_idx], X_use[test_idx]
            y_train = y_arr[train_idx]

            # Train fold model
            est = ExtraTreesRegressor(
                n_estimators=self.n_estimators // 2, # reduced for speed in OOF
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                max_features="sqrt",
                bootstrap=False,
                random_state=self.random_state + i,
                n_jobs=self.n_jobs
            )
            est.fit(X_train, y_train)
            oof_preds[test_idx] = est.predict(X_test)

        return oof_preds


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


def build_gamma_dataset(panel, feats, cfg, syms):
    """
    Build training dataset for Gamma Specialist.
    Returns: DataFrame with features and 'y_gamma' column.
    """
    tprint(f"Building Gamma Specialist dataset...")
    
    # 1. Generate gamma labels
    horizon = cfg.get("gamma_horizon", 6)
    tprint(f"  Computing gamma labels (horizon={horizon})...")
    gamma_labels = compute_gamma_labels(panel, feats, horizon=horizon)
    
    # 2. Build feature matrix
    tprint(f"  Building feature matrix ({len(GAMMA_FEATURE_KEYS)} features)...")
    data_list = []
    
    for sym in syms:
        if sym not in gamma_labels.columns:
            continue
        
        # Get gamma labels for this symbol
        y_sym = gamma_labels[sym].dropna()
        
        if len(y_sym) < 100:
            continue
        
        # Extract features for this symbol using reindexing
        valid_idx = y_sym.index.intersection(feats[GAMMA_FEATURE_KEYS[0]].index)
        if len(valid_idx) < 100: continue
        
        y_sym = y_sym.loc[valid_idx]

        # Check all features exist
        X_df_list = []
        valid_feats = True
        for k in GAMMA_FEATURE_KEYS:
            if k not in feats or sym not in feats[k].columns:
                valid_feats = False
                break
            X_df_list.append(feats[k][sym].reindex(valid_idx))
            
        if not valid_feats: continue

        X_sym = pd.concat(X_df_list, axis=1)
        X_sym.columns = GAMMA_FEATURE_KEYS

        # Combine
        combined = X_sym.copy()
        combined["y_gamma"] = y_sym.values
        combined["symbol"] = sym
        combined = combined.dropna()
        
        if len(combined) > 0:
            data_list.append(combined)
            
    if not data_list:
        tprint("  ERROR: No valid training data for Gamma Specialist")
        return None

    full_df = pd.concat(data_list)
    full_df.index.name = "ts"
    full_df = full_df.reset_index()
    
    tprint(f"  Gamma dataset: {len(full_df)} samples")
    return full_df


def train_gamma_from_dataset(dataset, cfg):
    """
    Train Gamma Specialist from pre-built dataset.
    """
    tprint("Training Gamma Specialist (ExtraTrees Regression) from dataset...")

    if dataset is None or dataset.empty:
        tprint("  ERROR: Dataset is empty.")
        return None

    X = dataset[GAMMA_FEATURE_KEYS]
    y = dataset["y_gamma"].values.astype(np.float32)
    
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


def train_gamma_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Gamma Specialist regression model (Legacy wrapper).
    """
    ds = build_gamma_dataset(panel, feats, cfg, syms)
    return train_gamma_from_dataset(ds, cfg)
