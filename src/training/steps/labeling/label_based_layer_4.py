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

    Normalizations (Approximate for OOF scale):
    - PnL: scaled by (1 / (N * 0.001)) -> roughly 1.0 if avg trade is 0.1%
    - Sortino: scaled by 1/3.0 -> 1.0 if Sortino is 3.0
    - DD: scaled by 1/0.2 -> 1.0 if DD is 20%
    """
    if len(returns) == 0:
        return 0.0

    # 1. Calculate PnL curve
    # Net returns = (Ret - Cost) * Size?
    # Or strictly: Trade Return = Size * Ret - Cost?
    # Usually: PnL = Size * (Ret - Cost) if cost is per dollar.
    # If cost is fixed bps per trade: PnL = Size * Ret - (Size > 0) * Cost?
    # Let's assume proportional cost: PnL = Size * (Ret - Cost)

    net_rets = returns - transaction_cost
    pnl_series = sizes * net_rets

    total_pnl = np.sum(pnl_series)

    # 2. Sortino
    # Downside deviation of pnl_series
    # We consider 0.0 (no trade) as neutral.
    # Downside = min(0, pnl).
    downside = np.minimum(0.0, pnl_series)
    downside_sq = downside ** 2
    mean_sq_down = np.mean(downside_sq)

    if mean_sq_down < 1e-12:
        sortino = 0.0
    else:
        # Annualization factor? Assuming 15m bars?
        # For utility ranking, raw ratio is fine.
        sortino = np.mean(pnl_series) / np.sqrt(mean_sq_down)
        # Clip Sortino to sane range [0, 10]
        sortino = np.clip(sortino, -2.0, 10.0)

    # 3. Max Drawdown
    equity = np.cumprod(1.0 + pnl_series)
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / (running_max + 1e-12))
    max_dd = np.max(dd)

    # 4. Normalize and Combine
    # Heuristic normalization targets
    norm_pnl = np.tanh(total_pnl * 2.0) # Map reasonable PnL to [-1, 1]
    # Actually, total PnL depends on N.
    # Let's use Average Trade PnL * 1000?
    avg_pnl = np.mean(pnl_series)
    # norm_pnl = np.tanh(avg_pnl * 5000) # 1bp * 5000 = 0.5. 2bp = 1.0.

    # Let's stick to the "Bounded" logic from thought process
    # Score = 0.7 * (PnL / Target) ...
    # Let's use Total Return %
    total_return = equity[-1] - 1.0
    # Cap return at 50% for normalization
    norm_return = np.clip(total_return / 0.5, -1.0, 1.0)

    # Sortino target 3.0
    norm_sortino = np.clip(sortino / 3.0, -1.0, 1.0)

    # DD target 0.2 (20%)
    norm_dd = np.clip(max_dd / 0.2, 0.0, 1.0)

    # Utility
    # We want to Maximize PnL, Maximize Sortino, Minimize DD
    utility = 0.7 * norm_return + 0.15 * norm_sortino + 0.15 * (1.0 - norm_dd)

    return float(utility)

# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_layer4_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute self-contained OHLCV-based regime features for Layer 4.
    """
    df = ohlcv.copy()
    try:
        from src.feature_generation.categories.layer3_specific_features import _compute_gate_regime_features
        features = _compute_gate_regime_features(df)
        for col in LAYER4_REGIME_FEATURES:
            if col not in features.columns:
                features[col] = np.nan
        return features[LAYER4_REGIME_FEATURES]
    except Exception:
        pass

    # Fallback implementation (Restored full logic)
    features = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']

    # Log returns
    log_ret = np.log(close / close.shift(1))

    # 1. Volatility Features
    features['rv_short'] = log_ret.rolling(window=12).std() * np.sqrt(12)
    rv_med = log_ret.rolling(window=48).std() * np.sqrt(48)
    features['rv_short_over_med'] = features['rv_short'] / (rv_med + 1e-8)

    tr = (high - low) / close
    atr_short = tr.rolling(window=12).mean()

    rv_long = log_ret.rolling(window=200).std()
    features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

    # 2. Trend Strength Features
    log_price = np.log(close)
    features['slope_short'] = log_price.diff(12).abs()

    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(window=14).sum()
    plus_di = pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    features['adx_proxy'] = dx.rolling(window=14).mean()

    # 3. Momentum
    features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()
    features['snr'] = features['momentum_short'].abs() / (features['rv_short'] + 1e-8)

    # 4. Vol Spike / Large Candle timing
    rv_mean = features['rv_short'].rolling(window=100, min_periods=20).mean()
    rv_std = features['rv_short'].rolling(window=100, min_periods=20).std()
    rv_z = (features['rv_short'] - rv_mean) / (rv_std + 1e-8)

    is_vol_spike = (rv_z > 2.0).astype(int)
    int_index = pd.Series(np.arange(len(df)), index=df.index)
    last_spike_int_idx = int_index.where(is_vol_spike == 1).ffill()
    features['time_since_last_vol_spike'] = int_index - last_spike_int_idx
    features['time_since_last_vol_spike'] = features['time_since_last_vol_spike'].fillna(1000)

    candle_range = high - low
    is_large_candle = (candle_range > 2.5 * atr_short).astype(int)
    large_candle_int_idx = int_index.where(is_large_candle == 1).ffill()
    features['time_since_last_large_candle'] = int_index - large_candle_int_idx
    features['time_since_last_large_candle'] = features['time_since_last_large_candle'].fillna(1000)

    # 5. Advanced Regime Features
    chop_window = 20
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr_full = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_tr = tr_full.rolling(chop_window).sum()
    max_hi = high.rolling(chop_window).max()
    min_lo = low.rolling(chop_window).min()
    range_hl = max_hi - min_lo

    features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)

    vr_window = 50
    r_20 = log_ret.rolling(20).sum()
    r_10 = log_ret.rolling(10).sum()
    var_20 = r_20.rolling(vr_window).var()
    var_10 = r_10.rolling(vr_window).var()
    features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)

    # Permutation Entropy
    pe_window = 50
    pe_dim = 3
    pe_values = close.values
    pe_n = len(pe_values)

    if pe_n >= pe_window + pe_dim:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(pe_values, window_shape=pe_dim)
        except ImportError:
            shape = (pe_n - pe_dim + 1, pe_dim)
            strides = (pe_values.strides[0], pe_values.strides[0])
            windows = np.lib.stride_tricks.as_strided(pe_values, shape=shape, strides=strides)

        patterns = np.argsort(windows, axis=1)
        perms = list(itertools.permutations(range(pe_dim)))
        perm_to_code = {p: i for i, p in enumerate(perms)}
        codes = np.apply_along_axis(lambda x: perm_to_code[tuple(x)], 1, patterns)

        code_series = pd.Series(codes, index=df.index[pe_dim - 1:])

        def calc_ent(x):
            counts = np.unique(x, return_counts=True)[1]
            probs = counts / counts.sum()
            max_ent = np.log2(math.factorial(pe_dim))
            ent = entropy(probs, base=2)
            return ent / max_ent

        rolling_ent = code_series.rolling(pe_window).apply(calc_ent, raw=True)
        features['permutation_entropy'] = rolling_ent.reindex(df.index)
    else:
        features['permutation_entropy'] = np.nan

    # Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    else:
        features['hour_sin'] = 0.0
        features['hour_cos'] = 0.0
        features['is_weekend'] = 0

    # 6. Efficiency Ratio (Kaufman's)
    er_window = 10
    change = (close - close.shift(er_window)).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    features['efficiency_ratio'] = change / (volatility + 1e-8)

    for col in LAYER4_REGIME_FEATURES:
        if col not in features.columns:
            features[col] = np.nan

    return features[LAYER4_REGIME_FEATURES]

# ---------------------------------------------------------------------------
# Layer 4 Risk Filter
# ---------------------------------------------------------------------------

class Layer4RiskFilter:
    """
    Layer 4 Risk Filter & Optimizer.
    
    Selects best model (ET, Ridge, LGBM) and optimizes HPO for
    custom portfolio utility (PnL, Sortino, DD) under quadratic sizing.
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
        n_trials: int = 20, # HPO trials
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
        self._l3_threshold: Optional[float] = None

    def fit(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        l3_probs: pd.Series,
        returns: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> 'Layer4RiskFilter':
        """
        Train Layer 4 with Model Race and HPO.
        """
        tprint_info(">>> Training Layer 4 Risk Filter (Race + HPO)...")

        # 1. Filter Data (Top L3 Quantile)
        l3_arr = pd.to_numeric(l3_probs, errors='coerce').values
        finite_mask = np.isfinite(l3_arr)
        if not np.any(finite_mask):
            tprint_warning("No finite L3 probs.")
            return self

        l3_threshold = _l3_threshold_from_keep_fraction(l3_arr, self.l3_keep_fraction)
        self._l3_threshold = l3_threshold
        training_mask = finite_mask & (l3_arr >= l3_threshold)
        
        n_train = int(np.sum(training_mask))
        if n_train < 100:
            tprint_warning(f"Too few samples ({n_train}). Skipping L4.")
            return self

        X_sub = X.loc[training_mask].copy()
        y_sub = y_true.loc[training_mask].values.astype(int)
        r_sub = returns.loc[training_mask].values.astype(float)
        
        # Handle sample weights
        if sample_weight is not None:
            w_sub = sample_weight.loc[training_mask].values
        else:
            w_sub = np.ones(len(y_sub))

        self.feature_names = X_sub.columns.tolist()

        # 2. Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        X_imp = self.imputer.fit_transform(X_sub)
        X_scaled = self.scaler.fit_transform(X_imp)

        # 3. Model Race
        tprint_info("   Running Model Race (ET vs Ridge vs LGBM)...")
        winner, best_score = self._run_model_race(X_scaled, y_sub, r_sub, w_sub)
        self.winning_model_type_ = winner
        tprint_success(f"   Race Winner: {winner} (Utility: {best_score:.4f})")

        # 4. HPO
        tprint_info(f"   Running HPO for {winner} ({self.n_trials} trials)...")
        best_params = self._run_hpo(winner, X_scaled, y_sub, r_sub, w_sub)
        self.best_params_ = best_params
        tprint_info(f"   Best Params: {best_params}")

        # 5. Final Fit
        tprint_info("   Fitting Final Model...")
        self.model = self._build_model(winner, best_params)
        self.model.fit(X_scaled, y_sub, sample_weight=w_sub)
        
        self._is_fitted = True
        return self

    def _run_model_race(
        self, X: np.ndarray, y: np.ndarray, r: np.ndarray, w: np.ndarray
    ) -> Tuple[str, float]:
        """Compare default models using TimeSeriesSplit and Custom Utility."""
        candidates = ['extratrees', 'ridge', 'lgbm']
        scores = {}
        
        # 3-fold TS split for evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for cand in candidates:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                r_val = r[val_idx]
                w_tr = w[train_idx]
                
                model = self._build_model(cand, default=True)
                try:
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    probs = model.predict_proba(X_val)[:, 1]

                    sizes = sizing_function(probs, self.sizing_threshold, self.sizing_gamma)
                    util = calculate_portfolio_utility(r_val, sizes)
                    fold_scores.append(util)
                except Exception as e:
                    tprint_error(f"Race failed for {cand}: {e}")
                    fold_scores.append(-1.0)
            
            avg_score = np.mean(fold_scores)
            scores[cand] = avg_score
            tprint_info(f"      {cand}: {avg_score:.4f}")

        best_cand = max(scores, key=scores.get)
        return best_cand, scores[best_cand]

    def _run_hpo(
        self, model_type: str, X: np.ndarray, y: np.ndarray, r: np.ndarray, w: np.ndarray
    ) -> Dict[str, Any]:
        """Run Optuna HPO."""

        # Use a single validation split for HPO speed (last 20%)
        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]
        r_val = r[split_idx:]
        w_tr = w[:split_idx]

        def objective(trial):
            params = {}
            if model_type == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                }
            elif model_type == 'extratrees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
            elif model_type == 'ridge':
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0), # ElasticNet
                }

            try:
                model = self._build_model(model_type, params)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                probs = model.predict_proba(X_val)[:, 1]
                sizes = sizing_function(probs, self.sizing_threshold, self.sizing_gamma)
                score = calculate_portfolio_utility(r_val, sizes)
                return score
            except Exception:
                return -10.0

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def _build_model(self, model_type: str, params: Optional[Dict[str, Any]] = None, default: bool = False) -> Any:
        if params is None:
            params = {}

        if model_type == 'lgbm':
            p = params.copy() if not default else {
                'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5
            }
            p['random_state'] = self.random_state
            p['n_jobs'] = 1 # Avoid threading issues in parallel HPO if any
            p['verbose'] = -1
            p['class_weight'] = self.class_weight
            return lgb.LGBMClassifier(**p)

        elif model_type == 'extratrees':
            p = params.copy() if not default else {
                'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 20
            }
            p['random_state'] = self.random_state
            p['bootstrap'] = True
            p['class_weight'] = self.class_weight
            p['n_jobs'] = self.n_jobs
            return ExtraTreesClassifier(**p)

        elif model_type == 'ridge':
            # Use LogisticRegression with ElasticNet
            p = params.copy() if not default else {'C': 1.0, 'l1_ratio': 0.5}
            return LogisticRegression(
                penalty='elasticnet', solver='saga', max_iter=2000,
                class_weight=self.class_weight, random_state=self.random_state,
                n_jobs=1, **p
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            return np.full(len(X), 0.5)

        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, filepath: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'winning_model': self.winning_model_type_,
            'best_params': self.best_params_,
            'config': {
                'sizing_threshold': self.sizing_threshold,
                'sizing_gamma': self.sizing_gamma
            }
        }, filepath)
        tprint_success(f"Layer 4 model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Layer4RiskFilter':
        data = joblib.load(filepath)
        cfg = data.get('config', {})
        obj = cls(
            sizing_threshold=cfg.get('sizing_threshold', 0.5),
            sizing_gamma=cfg.get('sizing_gamma', 2.0)
        )
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.imputer = data['imputer']
        obj.feature_names = data['feature_names']
        obj.winning_model_type_ = data.get('winning_model')
        obj.best_params_ = data.get('best_params')
        obj._is_fitted = True
        return obj

# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'target',
    return_col: str = 'realized_return',
    l3_quantile_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    tprint_info(">>> Running Layer 4 OOF Training & Optimization...")
    cfg = config or {}
    
    # 1. Regime Features
    regime_features = compute_layer4_regime_features(market_data)
    common_idx = oof_df.index.intersection(regime_features.index)
    
    X = regime_features.loc[common_idx]
    oof_aligned = oof_df.loc[common_idx]
    
    # 2. Add L3 Probs as Feature
    l3_probs = pd.to_numeric(oof_aligned[l3_prob_col], errors='coerce')
    y_true = pd.to_numeric(oof_aligned[target_col], errors='coerce')
    returns = pd.to_numeric(oof_aligned[return_col], errors='coerce')
    
    X = X.copy()
    X['l3_prob'] = l3_probs
    X['l3_lag'] = l3_probs.ewm(span=5, adjust=False).mean()

    # 3. OOF CV
    from src.utils.purged_kfold import PurgedKFoldTime
    cv = PurgedKFoldTime(n_splits=n_folds, purge=pd.Timedelta(minutes=60))
    splits = list(cv.split(X))
    
    l4_oof_probs = np.full(len(common_idx), np.nan)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        tprint_info(f"   Fold {fold_idx + 1}/{n_folds}...")
        
        # Prepare Fold Data
        X_train = X.iloc[train_idx]
        y_train = y_true.iloc[train_idx]
        l3_train = l3_probs.iloc[train_idx]
        ret_train = returns.iloc[train_idx]
        
        X_val = X.iloc[val_idx]

        # Valid mask
        mask = np.isfinite(y_train) & np.isfinite(ret_train)
        if mask.sum() < 50:
            continue
            
        l4_model = Layer4RiskFilter(
            n_trials=15, # Reduced trials for OOF speed
            l3_keep_fraction=float(cfg.get('layer4_quantile_threshold', 0.6))
        )

        l4_model.fit(X_train[mask], y_train[mask], l3_train[mask], ret_train[mask])
        l4_oof_probs[val_idx] = l4_model.predict_proba(X_val)

    # 4. Final Output
    oof_aligned = oof_aligned.copy()
    oof_aligned['layer4_prob'] = l4_oof_probs
    
    metrics = {
        'l4_oof_coverage': float(np.mean(np.isfinite(l4_oof_probs))),
        'l4_oof_mean': float(np.nanmean(l4_oof_probs))
    }
    
    return oof_aligned, metrics
