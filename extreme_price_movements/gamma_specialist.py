"""
Gamma Specialist: GMM for current volatility regime detection.

Uses GMM clustering to identify current volatility regimes (low/medium/high)
for dynamic risk adjustment (stop-loss, take-profit, position sizing).
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.utils import tprint


# Feature set for volatility regime detection (reduced to 9 core features)
# Organized by concept for GMM clustering
GAMMA_FEATURE_KEYS = [
    # 1. Core Volatility Level (absolute)
    "atr_pct",        # ATR as % - main volatility measure
    
    # 2. Volatility Regime (relative to history)  
    "vol_z",          # Z-score of current vol vs history
    
    # 3. Volatility Dynamics (rate of change)
    "vol_expansion_ratio",  # Recent vs historical vol (expansion signal)
    
    # 4. Vol-of-Vol (regime change indicator)
    "vov_ratio",      # Vol of vol - captures regime transitions
    
    # 5. Price Range (intensity)
    "range_24h_pct",  # 24h high-low range %
    
    # 6. Realized Volatility (multiple horizons, keep one)
    "rv_24h",         # 24h realized vol (most stable)
    
    # 7. Jump/Shock Detection
    "jump_rate_10h",  # Jump frequency - captures spike events
    
    # 8. Volatility Compression (mean-reversion signal)
    "vol_compression", # Compression before expansion
    
    # 9. Market Context
    "skew",           # Return skew - asymmetric volatility
]

# Volatility regime labels (for semantic meaning)
VOLATILITY_REGIMES = {
    0: "low",      # Low volatility regime
    1: "medium",   # Medium volatility regime  
    2: "high",     # High volatility regime
}


class GammaGMM:
    """
    GMM for volatility regime detection.
    Clusters current volatility features into regimes.
    """
    
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.scaler = None
        self.selected_features_ = GAMMA_FEATURE_KEYS
        self.regime_means_ = None  # Mean volatility by regime
        
    def fit(self, X, sample_weight=None):
        """Fit the GMM model on volatility features."""
        tprint(f"  GammaGMM: Fitting GMM with {self.n_components} components...")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=200,
            n_init=3,
            random_state=self.random_state,
            verbose=0
        )
        self.gmm.fit(X_scaled)
        
        # Compute mean volatility by regime (for semantic ordering)
        # We'll use the mean of atr_pct (first feature) within each cluster
        if "atr_pct" in X.columns:
            atr_idx = list(X.columns).index("atr_pct")
            self.regime_means_ = self.gmm.means_[:, atr_idx]
        else:
            self.regime_means_ = np.mean(X_scaled, axis=1)
        
        # Sort regimes by volatility (low to high)
        self.regime_order_ = np.argsort(self.regime_means_)
        self.regime_mapping_ = {old: new for new, old in enumerate(self.regime_order_)}
        
        tprint(f"  GammaGMM: Fitted. Regime means (ordered): {self.regime_means_[self.regime_order_]}")
        
        return self
    
    def predict(self, X):
        """Predict volatility regime (0=low, 1=medium, 2=high)."""
        X_scaled = self.scaler.transform(X)
        # Map to ordered regimes
        predictions = self.gmm.predict(X_scaled)
        return np.array([self.regime_mapping_.get(p, p) for p in predictions])
    
    def predict_proba(self, X):
        """Predict regime probabilities."""
        X_scaled = self.scaler.transform(X)
        probs = self.gmm.predict_proba(X_scaled)
        # Reorder columns to match regime mapping
        new_probs = np.zeros_like(probs)
        for old, new in self.regime_mapping_.items():
            new_probs[:, new] = probs[:, old]
        return new_probs
    
    def score_samples(self, X):
        """Get log-likelihood scores (used like other specialist scores)."""
        X_scaled = self.scaler.transform(X)
        # Return probability of high volatility regime (regime 2)
        probs = self.predict_proba(X)
        return probs[:, 2]  # High volatility probability


def build_gamma_dataset(panel, feats, cfg, syms):
    """
    Build training dataset for Gamma Specialist (Volatility Regime).
    Returns: DataFrame with features.
    """
    tprint(f"Building Gamma Specialist dataset...")
    
    data_list = []
    for sym in syms:
        # Check all features exist
        X_df_list = []
        valid_feats = True
        for k in GAMMA_FEATURE_KEYS:
            if k not in feats or sym not in feats[k].columns:
                valid_feats = False
                break
            X_df_list.append(feats[k][sym])
            
        if not valid_feats: continue

        X_sym = pd.concat(X_df_list, axis=1)
        X_sym.columns = GAMMA_FEATURE_KEYS
        X_sym["symbol"] = sym

        # Drop NaNs
        X_sym = X_sym.dropna()

        if len(X_sym) > 0:
            data_list.append(X_sym)
            
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
    Train Gamma Specialist GMM from pre-built dataset.
    Uses current volatility features to cluster into regimes.
    """
    tprint("Training Gamma Specialist (GMM for Volatility Regime) from dataset...")

    if dataset is None or dataset.empty:
        tprint("  ERROR: Dataset is empty.")
        return None

    X = dataset[GAMMA_FEATURE_KEYS].copy()
    
    # Handle missing values
    X = X.fillna(0.0)
    
    tprint(f"  Training data: {len(X)} samples, {X.shape[1]} features")
    
    # Subsample for GMM fitting (GMM on millions of rows is slow)
    max_gamma_samples = cfg.get("gamma_max_gmm_samples", 300_000)
    if len(X) > max_gamma_samples:
        rng = np.random.RandomState(cfg.get("random_state", 42))
        idx_sub = rng.choice(len(X), max_gamma_samples, replace=False)
        X_fit = X.iloc[idx_sub].reset_index(drop=True)
        tprint(f"  Subsampled {max_gamma_samples} / {len(X)} for GMM fitting")
    else:
        X_fit = X
    
    # Fit GMM
    n_components = cfg.get("gamma_n_components", 3)
    model = GammaGMM(n_components=n_components, random_state=cfg.get("random_state", 42))
    model.fit(X_fit)
    
    # Generate scores on full dataset for validation
    all_scores = model.score_samples(X)
    tprint(f"  Gamma regime scores: mean={all_scores.mean():.3f}, std={all_scores.std():.3f}")
    
    return model


# ============================================================================
# DEPRECATED: Old ExtraTrees-based Gamma Model
# ============================================================================
# The GammaModel class and related functions have been replaced by GammaGMM.
# Keeping stub for backward compatibility during migration.

class GammaModel:
    """DEPRECATED: Use GammaGMM instead."""
    pass


def train_gamma_specialist(panel, feats, cfg, syms, ts_end):
    """
    Train Gamma Specialist GMM model (Legacy wrapper).
    """
    ds = build_gamma_dataset(panel, feats, cfg, syms)
    return train_gamma_from_dataset(ds, cfg)
