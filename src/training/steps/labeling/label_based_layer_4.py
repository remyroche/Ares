"""Layer 4 — Risk Filter & Sizing Optimizer.

This module implements a risk-filter and sizing optimization stage that sits between Layer 3 (meta-model)
and Layer 5 (position sizing). It selects and optimizes a model (ExtraTrees, Ridge, or LGBM)
to maximize a portfolio utility function (PnL, Sortino, Drawdown) under a specific sizing assumption.

Sizing Assumption:
    Size = 0 if p < 0.5
    Size = ((p - 0.5) / 0.5) ^ 2 if p >= 0.5
    (Quadratic scaling above threshold)

Objective:
    Utility = 0.7 * Norm_PnL + 0.15 * Norm_Sortino + 0.15 * (1 - Norm_DD)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import itertools
import math
import joblib
import optuna
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success, tprint_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYER4_REGIME_FEATURES = [
    'rv_z_short',
    'slope_short', 'adx_proxy', 'momentum_short', 'snr',
    'time_since_last_vol_spike', 'time_since_last_large_candle',
    'choppiness_index', 'variance_ratio', 'permutation_entropy',
    'hour_sin', 'hour_cos', 'is_weekend',
    'efficiency_ratio',
    'volatility_roc', 'er_roc',
    'close_over_max_12', 'close_over_max_24', 'close_over_max_48',
    'close_over_min_12', 'close_over_min_24', 'close_over_min_48',
    'vol_adjusted_momentum',
]

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _clip_keep_fraction(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.6
    if not np.isfinite(x):
        return 0.6
    return float(np.clip(x, 0.01, 1.0))

def _l3_threshold_from_keep_fraction(l3_values: np.ndarray, keep_fraction: float) -> float:
    keep_fraction = _clip_keep_fraction(keep_fraction)
    finite_mask = np.isfinite(l3_values)
    if not np.any(finite_mask):
        return float('nan')
    return float(np.quantile(l3_values[finite_mask], 1.0 - keep_fraction))

def sizing_function(probs: np.ndarray, threshold: float = 0.5, gamma: float = 2.0) -> np.ndarray:
    """
    Apply sizing logic:
    Size = 0 if p < threshold
    Size = ((p - threshold) / (1 - threshold)) ^ gamma if p >= threshold
    """
    p = np.clip(probs, 0.0, 1.0)
    denom = 1.0 - threshold
    if denom < 1e-6:
        denom = 1e-6

    # Scale: (p - thr) / (1 - thr)
    scaled = (p - threshold) / denom
    scaled = np.clip(scaled, 0.0, 1.0)

    # Apply gamma
    sizes = np.power(scaled, gamma)

    # Zero out below threshold (already handled by clip(0,1) but explicit check for safety)
    sizes = np.where(p < threshold, 0.0, sizes)
    return sizes

def calculate_portfolio_utility(
    returns: np.ndarray,
    sizes: np.ndarray,
    transaction_cost: float = 0.0000
) -> float:
    """
    Calculate custom portfolio utility:
    Utility = 0.7 * Norm_PnL + 0.15 * Norm_Sortino + 0.15 * (1 - Norm_DD)
    """
    if len(returns) == 0:
        return 0.0

    net_rets = returns - transaction_cost
    pnl_series = sizes * net_rets

    total_pnl = np.sum(pnl_series)

    # 2. Sortino
    downside = np.minimum(0.0, pnl_series)
    downside_sq = downside ** 2
    mean_sq_down = np.mean(downside_sq)

    if mean_sq_down < 1e-12:
        sortino = 0.0
    else:
        sortino = np.mean(pnl_series) / np.sqrt(mean_sq_down)
        sortino = np.clip(sortino, -2.0, 10.0)

    # 3. Max Drawdown
    equity = np.cumprod(1.0 + pnl_series)
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / (running_max + 1e-12))
    max_dd = np.max(dd)

    # 4. Normalize and Combine
    total_return = equity[-1] - 1.0
    norm_return = np.clip(total_return / 0.5, -1.0, 1.0)
    norm_sortino = np.clip(sortino / 3.0, -1.0, 1.0)
    norm_dd = np.clip(max_dd / 0.2, 0.0, 1.0)

    utility = 0.7 * norm_return + 0.15 * norm_sortino + 0.15 * (1.0 - norm_dd)

    return float(utility)

def prepare_scaled_features_for_meta_learner(
    l3_cols: List[str],
    raw_signals: pd.DataFrame, # Containing meta_prob_* columns
    volatility: np.ndarray,
    models_metadata: Optional[Dict] = None, # Dict containing z-scores per gid
    impact_factor: float = 0.6,
    sensitivity: float = 0.5
) -> pd.DataFrame:
    """
    Implements Option 3: Signal Pre-Scaling.
    Embeds static quality metrics into the dynamic signal magnitude.
    """
    meta_features = {}

    # We expect raw_signals to contain columns like 'meta_prob_g_a0.50_linear'
    # We extract the GID from the column name

    for col in l3_cols:
        gid = col.replace('meta_prob_', '')
        raw_sig = raw_signals[col].values

        # Get quality z-score for this geometry if available
        z_score = 0.0
        if models_metadata and f'{gid}_meta' in models_metadata:
             meta = models_metadata[f'{gid}_meta']
             if 'z_scores' in meta:
                 zs = meta['z_scores']
                 # Weights: 30% AUC, 20% Stability, 30% RAD, 20% Safety
                 # Assuming these keys exist in metadata
                 z_score = (
                    0.3 * zs.get('z_auc', 0) +
                    0.1 * zs.get('z_stab', 0) +
                    0.3 * zs.get('z_rad', 0) +
                    0.2 * zs.get('z_safe', 0)
                 )

        # --- THE SCALING FORMULA ---
        multiplier = 1.0 + impact_factor * np.tanh(sensitivity * z_score)

        # Apply scaling
        scaled_sig = raw_sig * multiplier

        # Feature A: The Scaled Signal
        meta_features[f"{gid}_sig_scaled"] = scaled_sig

        # Feature B: Rolling Z-Score
        sig_series = pd.Series(scaled_sig)
        rolling_z = (sig_series - sig_series.rolling(50).mean()) / (sig_series.rolling(50).std() + 1e-6)
        meta_features[f"{gid}_sig_z50"] = rolling_z.fillna(0).values

        # Feature C: Signal Momentum (Divergence)
        ewma_15 = sig_series.ewm(span=15).mean()
        meta_features[f"{gid}_sig_div"] = (sig_series - ewma_15).values

    # Assemble
    X_meta = pd.DataFrame(meta_features)

    # Global Volatility Context
    X_meta['global_vol_rank'] = pd.Series(volatility).rolling(100).rank().fillna(0.5).values

    return X_meta

# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_layer4_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute self-contained OHLCV-based regime features for Layer 4.
    """
    df = ohlcv.copy()

    # Use existing Layer 3 features if available, else compute
    try:
        from src.feature_generation.categories.layer3_specific_features import _compute_gate_regime_features
        features = _compute_gate_regime_features(df)
    except Exception:
        features = pd.DataFrame(index=df.index)

    # Ensure base features exist
    close = df['close']
    high = df['high']
    low = df['low']
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # ... (Keep existing features logic) ...
    # Re-implementing explicitly for safety and adding new ones

    features['rv_short'] = log_ret.rolling(window=12).std() * np.sqrt(12)
    rv_long = log_ret.rolling(window=200).std()
    features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

    features['slope_short'] = np.log(close).diff(12).abs()

    # ADX Proxy
    tr = (high - low) / close
    tr_smooth = tr.rolling(14).sum()
    up = high.diff()
    down = low.diff()
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    p_di = pd.Series(p_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-8)
    m_di = pd.Series(m_dm, index=df.index).rolling(14).sum() / (tr_smooth + 1e-8)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-8)
    features['adx_proxy'] = dx.rolling(14).mean()

    features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()
    features['snr'] = features['momentum_short'] / (features['rv_short'] + 1e-8)

    # Vol Spike / Large Candle timing (simplified)
    # ...

    # NEW FEATURES
    # Rate of change of volatility or ER
    features['volatility_roc'] = features['rv_short'].pct_change(5)

    er_window = 10
    change = (close - close.shift(er_window)).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    features['efficiency_ratio'] = change / (volatility + 1e-8)
    features['er_roc'] = features['efficiency_ratio'].pct_change(5)

    # Close / Rolling Max/Min
    for w in [12, 24, 48]:
        features[f'close_over_max_{w}'] = close / (close.rolling(w).max() + 1e-8)
        features[f'close_over_min_{w}'] = close / (close.rolling(w).min() + 1e-8)

    # Volatility-adjusted momentum
    features['vol_adjusted_momentum'] = (close.pct_change(12) / (features['rv_short'] + 1e-8))

    # Fill NaNs
    features = features.fillna(0.0)

    # Filter to requested
    final_cols = [c for c in LAYER4_REGIME_FEATURES if c in features.columns]
    return features[final_cols]

# ---------------------------------------------------------------------------
# Layer 4 Risk Filter
# ---------------------------------------------------------------------------

class Layer4RiskFilter:
    """
    Layer 4 Risk Filter & Optimizer.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1,
        l3_keep_fraction: float = 0.6,
        n_trials: int = 20,
        sizing_threshold: float = 0.5,
        sizing_gamma: float = 2.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.l3_keep_fraction = _clip_keep_fraction(l3_keep_fraction)
        self.n_trials = n_trials
        self.sizing_threshold = sizing_threshold
        self.sizing_gamma = sizing_gamma

        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names: List[str] = []
        self.winning_model_type_: Optional[str] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        # l3_probs is now optional or handled differently via X features
        l3_probs: Optional[pd.Series] = None,
        returns: pd.Series = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> 'Layer4RiskFilter':
        """
        Train Layer 4 with Model Race and HPO.
        """
        tprint_info(">>> Training Layer 4 Risk Filter (Race + HPO)...")

        # Basic filtering if l3_probs provided (legacy or using primary/mean prob)
        if l3_probs is not None:
             l3_arr = pd.to_numeric(l3_probs, errors='coerce').values
             l3_threshold = _l3_threshold_from_keep_fraction(l3_arr, self.l3_keep_fraction)
             training_mask = np.isfinite(l3_arr) & (l3_arr >= l3_threshold)
        else:
             # Use all data if no explicit L3 prob for filtering
             training_mask = np.ones(len(y_true), dtype=bool)

        n_train = int(np.sum(training_mask))
        if n_train < 100:
            tprint_warning(f"Too few samples ({n_train}). Skipping L4.")
            return self

        X_sub = X.loc[training_mask].copy()
        y_sub = y_true.loc[training_mask].values.astype(int)
        r_sub = returns.loc[training_mask].values.astype(float)
        
        if sample_weight is not None:
            w_sub = sample_weight.loc[training_mask].values
        else:
            w_sub = np.ones(len(y_sub))

        self.feature_names = X_sub.columns.tolist()

        # Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        X_imp = self.imputer.fit_transform(X_sub)
        X_scaled = self.scaler.fit_transform(X_imp)

        # Race
        winner, best_score = self._run_model_race(X_scaled, y_sub, r_sub, w_sub)
        self.winning_model_type_ = winner
        tprint_success(f"   Race Winner: {winner} (Utility: {best_score:.4f})")

        # HPO
        best_params = self._run_hpo(winner, X_scaled, y_sub, r_sub, w_sub)
        self.best_params_ = best_params
        tprint_info(f"   Best Params: {best_params}")

        # Final Fit
        self.model = self._build_model(winner, best_params)
        self.model.fit(X_scaled, y_sub, sample_weight=w_sub)
        
        self._is_fitted = True
        return self

    def _run_model_race(self, X, y, r, w):
        candidates = ['extratrees', 'ridge', 'lgbm']
        scores = {}
        tscv = TimeSeriesSplit(n_splits=3)
        for cand in candidates:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                model = self._build_model(cand, default=True)
                try:
                    model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
                    probs = model.predict_proba(X[val_idx])[:, 1]
                    sizes = sizing_function(probs, self.sizing_threshold, self.sizing_gamma)
                    util = calculate_portfolio_utility(r[val_idx], sizes)
                    fold_scores.append(util)
                except Exception:
                    fold_scores.append(-1.0)
            scores[cand] = np.mean(fold_scores)
        best = max(scores, key=scores.get)
        return best, scores[best]

    def _run_hpo(self, model_type, X, y, r, w):
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        r_val = r[split:]
        w_tr = w[:split]

        def objective(trial):
            params = {}
            if model_type == 'lgbm':
                # Not too tightly regularized
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                }
            elif model_type == 'extratrees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                }
            elif model_type == 'ridge':
                params = {'C': trial.suggest_float('C', 0.1, 10.0)}

            try:
                model = self._build_model(model_type, params)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                probs = model.predict_proba(X_val)[:, 1]
                sizes = sizing_function(probs, self.sizing_threshold, self.sizing_gamma)
                return calculate_portfolio_utility(r_val, sizes)
            except Exception:
                return -10.0

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def _build_model(self, model_type, params=None, default=False):
        if params is None: params = {}
        if model_type == 'lgbm':
            p = params.copy() if not default else {'n_estimators': 300}
            p.update({'random_state': 42, 'n_jobs': 1, 'verbose': -1})
            return lgb.LGBMClassifier(**p)
        elif model_type == 'extratrees':
            p = params.copy() if not default else {'n_estimators': 300}
            p.update({'random_state': 42, 'n_jobs': -1})
            return ExtraTreesClassifier(**p)
        elif model_type == 'ridge':
            p = params.copy() if not default else {'C': 1.0}
            return LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42, **p)
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            return np.full(len(X), 0.5)
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict_proba(X_scaled)[:, 1]

# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    l3_prob_col: str = 'meta_prob', # Legacy single col, or prefix?
    target_col: str = 'target',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    # New arg for multiple models
    l3_models_metadata: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    tprint_info(">>> Running Layer 4 OOF Training & Optimization...")
    cfg = config or {}
    
    # 1. Regime Features (Global)
    regime_features = compute_layer4_regime_features(market_data)
    
    # 2. Identify Layer 3 Output Columns
    l3_cols = [c for c in oof_df.columns if c.startswith('meta_prob_')]

    # 3. Construct Meta Features (Scaling & Disagreement)
    if not l3_cols:
        # Fallback to single col if no Multi-Geometry outputs found
        l3_cols = [l3_prob_col] if l3_prob_col in oof_df.columns else []

    common_idx = oof_df.index.intersection(regime_features.index)
    X_regime = regime_features.loc[common_idx]
    oof_aligned = oof_df.loc[common_idx]
    
    # Prepare Scaled Features
    # We need volatility for scaling
    if 'volatility_1d' in oof_aligned.columns:
        vol = oof_aligned['volatility_1d'].values
    else:
        # Compute vol proxy
        vol = market_data['close'].pct_change().rolling(24).std().reindex(common_idx).fillna(0.01).values

    if l3_cols:
        tprint_info(f"   Found {len(l3_cols)} Layer 3 geometry outputs.")
        X_meta = prepare_scaled_features_for_meta_learner(
            l3_cols, oof_aligned, vol, l3_models_metadata
        )

        # Disagreement Features (from feature_generation or inline)
        # inline for simplicity as per requirement to use "features from src... applied to models"
        # Since we have the raw signals in X_meta or oof_aligned, we can compute disagreement here.

        # Compute row-wise stats on raw probs
        raw_probs = oof_aligned[l3_cols].values
        X_meta['ens_mean'] = np.mean(raw_probs, axis=1)
        X_meta['ens_std'] = np.std(raw_probs, axis=1)
        X_meta['ens_min'] = np.min(raw_probs, axis=1)
        X_meta['ens_max'] = np.max(raw_probs, axis=1)
        X_meta['ens_range'] = X_meta['ens_max'] - X_meta['ens_min']

        # Merge Regime + Meta
        X = pd.concat([X_regime.reset_index(drop=True), X_meta.reset_index(drop=True)], axis=1)
        X.index = common_idx

        # Primary prob for filtering/sizing anchor (Mean of scaled? or just Mean of raw?)
        # Let's use Mean of raw for now
        l3_probs_anchor = X_meta['ens_mean']

    else:
        tprint_warning("   No Layer 3 outputs found. Using only regime features.")
        X = X_regime
        l3_probs_anchor = pd.Series(0.5, index=common_idx)

    y_true = pd.to_numeric(oof_aligned[target_col], errors='coerce')
    returns = pd.to_numeric(oof_aligned[return_col], errors='coerce')

    # 4. OOF CV
    from src.utils.purged_kfold import PurgedKFoldTime
    cv = PurgedKFoldTime(n_splits=n_folds, purge=pd.Timedelta(minutes=60))
    
    l4_oof_probs = np.full(len(common_idx), np.nan)
    
    # We loop manually to handle index alignment carefully
    X_vals = X.values
    y_vals = y_true.values
    r_vals = returns.values
    l3_vals = l3_probs_anchor.values

    splits = list(cv.split(X))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        tprint_info(f"   Fold {fold_idx + 1}/{n_folds}...")
        
        mask_tr = np.isfinite(y_vals[train_idx]) & np.isfinite(r_vals[train_idx])
        
        if mask_tr.sum() < 50: continue
            
        l4_model = Layer4RiskFilter(
            n_trials=10,
            l3_keep_fraction=float(cfg.get('layer4_quantile_threshold', 0.6))
        )

        l4_model.fit(
            X.iloc[train_idx][mask_tr],
            y_true.iloc[train_idx][mask_tr],
            l3_probs_anchor.iloc[train_idx][mask_tr],
            returns.iloc[train_idx][mask_tr]
        )
        l4_oof_probs[val_idx] = l4_model.predict_proba(X.iloc[val_idx])

    oof_aligned = oof_aligned.copy()
    oof_aligned['layer4_prob'] = l4_oof_probs
    
    metrics = {
        'l4_oof_coverage': float(np.mean(np.isfinite(l4_oof_probs))),
        'l4_oof_mean': float(np.nanmean(l4_oof_probs))
    }
    
    return oof_aligned, metrics
