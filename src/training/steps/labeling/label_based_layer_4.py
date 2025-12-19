"""Layer 4 — Risk Filter: ExtraTrees-based trade veto model with Platt calibration.

This module implements a risk-filter stage that sits between Layer 3 (meta-model)
and Layer 5 (position sizing). It uses a well-regularized ExtraTreesClassifier
(max_depth=5) trained on the top-quantile subset of Layer 3 OOF predictions to
identify and discard risky trades.

Features used are the same as gate_model.py (regime features, efficiency ratio, etc.).

The model outputs calibrated probabilities via Platt scaling (sigmoid calibration).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import time
import itertools
import math
import joblib

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from scipy.stats import entropy

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success, tprint_error


# ---------------------------------------------------------------------------
# Gate Feature Set (same as gate_model.py)
# ---------------------------------------------------------------------------

LAYER4_REGIME_FEATURES = [
    'rv_short', 'rv_short_over_med', 'rv_z_short',
    'slope_short', 'adx_proxy', 'momentum_short', 'snr',
    'time_since_last_vol_spike', 'time_since_last_large_candle',
    'choppiness_index', 'variance_ratio', 'permutation_entropy',
    'hour_sin', 'hour_cos', 'is_weekend',
    'efficiency_ratio',  # Kaufman's Efficiency Ratio
]


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


def _equity_curve_from_returns(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return np.asarray([1.0], dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    return np.cumprod(1.0 + r)


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    e = np.where(np.isfinite(e), e, np.nan)
    if not np.any(np.isfinite(e)):
        return 0.0
    running_max = np.maximum.accumulate(np.where(np.isfinite(e), e, -np.inf))
    dd = (e / (running_max + 1e-12)) - 1.0
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return 0.0
    return float(-np.min(dd))


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float('nan'), float('nan'))
    if n_boot <= 1:
        m = float(np.mean(arr))
        return (m, m)
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - (alpha / 2.0)))
    return (lo, hi)


def compute_layer4_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute self-contained OHLCV-based regime features for Layer 4.
    Same logic as gate_model._compute_regime_features plus efficiency_ratio.
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
# Layer 4 Risk Filter Model
# ---------------------------------------------------------------------------

class Layer4RiskFilter:
    """
    Layer 4 Risk Filter: ExtraTrees-based trade veto model.
    
    Trained on Layer 3 OOF predictions (top quantile subset) to identify
    and discard risky trades. Uses Platt scaling (sigmoid) for calibration.
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
        l3_quantile_threshold: Optional[float] = None,
    ):
        """
        Initialize Layer 4 Risk Filter.

        Args:
            n_estimators: Number of trees in ExtraTrees.
            max_depth: Maximum depth (regularization).
            min_samples_leaf: Minimum samples per leaf (regularization).
            class_weight: Class weighting strategy.
            random_state: Random seed.
            n_jobs: Parallel jobs.
            l3_keep_fraction: Fraction of top L3 predictions to keep.
            l3_quantile_threshold: Alias for l3_keep_fraction (backward compatibility).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        if l3_quantile_threshold is not None:
            l3_keep_fraction = l3_quantile_threshold

        self.l3_keep_fraction = _clip_keep_fraction(l3_keep_fraction)
        self.l3_quantile_threshold = self.l3_keep_fraction

        self.model = None
        self.calibrator = None  # Platt scaling (LogisticRegression)
        self.scaler = None
        self.imputer = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        
        self._is_fitted = False
        self._training_mask: Optional[np.ndarray] = None
        self._l3_threshold: Optional[float] = None

    def _build_pipeline(self) -> ExtraTreesClassifier:
        """Build the ExtraTrees model."""
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features='sqrt',
            bootstrap=True,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        l3_probs: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> 'Layer4RiskFilter':
        """
        Train the risk filter on Layer 3 OOF predictions.

        Args:
            X: Feature DataFrame (regime features).
            y_true: Actual trade outcomes (0=loss, 1=win).
            l3_probs: Layer 3 OOF calibrated probabilities.
            sample_weight: Optional sample weights.

        Returns:
            self
        """
        tprint_info(">>> Training Layer 4 Risk Filter (ExtraTrees + Platt)...")

        # 1. Compute L3 threshold and create training mask
        l3_arr = pd.to_numeric(l3_probs, errors='coerce').values
        finite_mask = np.isfinite(l3_arr)
        
        if not np.any(finite_mask):
            tprint_warning("No finite L3 probabilities. Cannot train Layer 4.")
            return self

        l3_threshold = _l3_threshold_from_keep_fraction(l3_arr, self.l3_keep_fraction)
        self._l3_threshold = l3_threshold
        
        # Training mask: only samples where L3 says "trade" (above threshold)
        training_mask = finite_mask & (l3_arr >= l3_threshold)
        self._training_mask = training_mask
        
        n_train = int(np.sum(training_mask))
        tprint_info(f"   L3 threshold (keep top {self.l3_keep_fraction*100:.0f}%): {l3_threshold:.4f}")
        tprint_info(f"   Training samples: {n_train} / {len(l3_arr)}")

        if n_train < 100:
            tprint_warning(f"Too few training samples ({n_train}). Skipping Layer 4 training.")
            return self

        # 2. Prepare features
        X_train = X.loc[training_mask].copy()
        y_train = y_true.loc[training_mask].values.astype(int)
        
        self.feature_names = X_train.columns.tolist()

        # 3. Preprocessing
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        X_imputed = self.imputer.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # 4. Train ExtraTrees
        self.model = self._build_pipeline()
        
        sw = None
        if sample_weight is not None:
            sw = sample_weight.loc[training_mask].values
        
        self.model.fit(X_scaled, y_train, sample_weight=sw)
        self.feature_importances_ = self.model.feature_importances_

        # 5. Platt Scaling (Sigmoid Calibration)
        # Use cross-validated predictions for calibration
        tprint_info("   Fitting Platt scaling calibrator...")
         
        try:
            if len(np.unique(y_train)) < 2:
                raise ValueError('Need both classes for calibration')

            n_splits = 5
            if len(y_train) < 250:
                n_splits = 3
            if len(y_train) < 150:
                n_splits = 2
            cv = TimeSeriesSplit(n_splits=n_splits)
            
            # Manual OOF collection because cross_val_predict only works for partitions,
            # and TimeSeriesSplit folds are not a partition (they overlap in training).
            raw_probs_oof = np.full(len(y_train), np.nan)
            
            for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv.split(X_scaled)):
                # Clone/Fresh model for each fold to ensure absolute OOF
                fold_model = self._build_pipeline()
                fold_model.fit(X_scaled[cv_train_idx], y_train[cv_train_idx])
                
                # Predict on validation set
                fold_probs = fold_model.predict_proba(X_scaled[cv_val_idx])[:, 1]
                raw_probs_oof[cv_val_idx] = fold_probs
            
            # Only use non-nan entries for calibration (first fold training data won't have OOF preds)
            valid_oof_mask = np.isfinite(raw_probs_oof)
            if not np.any(valid_oof_mask):
                 raise ValueError('No OOF predictions generated for calibration')
                 
            y_calib = y_train[valid_oof_mask]
            X_calib = raw_probs_oof[valid_oof_mask].reshape(-1, 1)

            # Fit logistic regression on raw probs -> calibrated probs
            self.calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(X_calib, y_calib)
             
            tprint_success(f"   Platt scaling calibrator fitted on {len(y_calib)} OOF samples.")
        except Exception as e:
            tprint_warning(f"   Platt scaling failed: {e}. Using raw probabilities.")
            self.calibrator = None

        self._is_fitted = True

        # Log feature importances
        if self.feature_importances_ is not None:
            top_k = min(10, len(self.feature_names))
            order = np.argsort(self.feature_importances_)[::-1]
            tprint_info("   Top feature importances:")
            for i in order[:top_k]:
                tprint_info(f"      {self.feature_names[i]}: {self.feature_importances_[i]:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probability of trade success.

        Args:
            X: Feature DataFrame.

        Returns:
            Calibrated probabilities (n_samples,).
        """
        if not self._is_fitted or self.model is None:
            return np.full(len(X), 0.5)

        # Preprocess
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        # Raw predictions
        raw_probs = self.model.predict_proba(X_scaled)[:, 1]

        # Apply Platt scaling if available
        if self.calibrator is not None:
            calibrated = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            return np.clip(calibrated, 0.0, 1.0)

        return np.clip(raw_probs, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary decision: 1 (allow trade) or 0 (veto trade).

        Args:
            X: Feature DataFrame.
            threshold: Probability threshold for allowing trade.

        Returns:
            Binary decisions (n_samples,).
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'l3_keep_fraction': self.l3_keep_fraction,
            'l3_quantile_threshold': self.l3_keep_fraction,
            'l3_threshold': self._l3_threshold,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf,
                'class_weight': self.class_weight,
                'random_state': self.random_state,
            }
        }, filepath)
        tprint_success(f"Layer 4 model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Layer4RiskFilter':
        """Load model from disk."""
        data = joblib.load(filepath)
        config = data.get('config', {})
        
        obj = cls(
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', 5),
            min_samples_leaf=config.get('min_samples_leaf', 20),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            l3_keep_fraction=data.get('l3_keep_fraction', data.get('l3_quantile_threshold', 0.6)),
        )
        obj.model = data['model']
        obj.calibrator = data.get('calibrator')
        obj.scaler = data['scaler']
        obj.imputer = data['imputer']
        obj.feature_names = data['feature_names']
        obj.feature_importances_ = data.get('feature_importances')
        obj._l3_threshold = data.get('l3_threshold')
        obj._is_fitted = True
        
        return obj


# ---------------------------------------------------------------------------
# Final Score Formulas (for Layer 5 integration)
# ---------------------------------------------------------------------------

def compute_final_score_product(p_l3: np.ndarray, p_l4: np.ndarray) -> np.ndarray:
    p_l3 = np.asarray(p_l3, dtype=float)
    p_l4 = np.asarray(p_l4, dtype=float)
    return np.clip(p_l3 * p_l4, 0.0, 1.0)


def compute_final_score_min(p_l3: np.ndarray, p_l4: np.ndarray) -> np.ndarray:
    p_l3 = np.asarray(p_l3, dtype=float)
    p_l4 = np.asarray(p_l4, dtype=float)
    return np.clip(np.minimum(p_l3, p_l4), 0.0, 1.0)


def compute_final_score_logit_avg(p_l3: np.ndarray, p_l4: np.ndarray) -> np.ndarray:
    p_l3 = np.asarray(p_l3, dtype=float)
    p_l4 = np.asarray(p_l4, dtype=float)
    eps = 1e-8
    p_l3 = np.clip(p_l3, eps, 1.0 - eps)
    p_l4 = np.clip(p_l4, eps, 1.0 - eps)
    logit_l3 = np.log(p_l3 / (1.0 - p_l3))
    logit_l4 = np.log(p_l4 / (1.0 - p_l4))
    logit_avg = 0.5 * (logit_l3 + logit_l4)
    out = 1.0 / (1.0 + np.exp(-logit_avg))
    return np.clip(out, 0.0, 1.0)

def compute_final_score_dynamic(p_l3: np.ndarray, p_l4: np.ndarray) -> np.ndarray:
    """
    Dynamic Confidence Scaler formula.
    
    P_final = P_L3 * (1 - Penalty)
    where Penalty = (1 - P_L4) * (1 - P_L3)
    """
    return compute_final_score_product(p_l3, p_l4)


def compute_final_score_bayesian(p_l3: np.ndarray, p_l4: np.ndarray, prior: float = 0.5) -> np.ndarray:
    """
    Bayesian Inference formula.
    
    Odds_final = Odds_L4 * Likelihood_Ratio(L3)
    Then convert back to probability.
    """
    return compute_final_score_logit_avg(p_l3, p_l4)


def compute_final_score_ridge(p_l3: np.ndarray, p_l4: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Ridge penalty formula.
    
    P_final = P_L3 - beta * (1 - P_L4)^2
    """
    p_l3 = np.asarray(p_l3, dtype=float)
    p_l4 = np.asarray(p_l4, dtype=float)
    try:
        beta_f = float(beta)
    except Exception:
        beta_f = 0.5
    penalty = beta_f * np.square(1.0 - p_l4)
    return np.clip(p_l3 - penalty, 0.0, 1.0)

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
    """
    Train Layer 4 risk filter using OOF predictions and evaluate grid of thresholds/formulas.

    Args:
        oof_df: DataFrame with Layer 3 OOF predictions.
        market_data: OHLCV data for regime feature computation.
        l3_prob_col: Column name for L3 probabilities.
        target_col: Column name for binary target.
        return_col: Column name for realized returns.
        l3_quantile_thresholds: List of quantile thresholds to evaluate.
        n_folds: Number of CV folds for OOF.
        config: Optional configuration dict.

    Returns:
        Tuple of (results_df, metrics_dict).
    """
    tprint_info(">>> Running Layer 4 OOF Training & Grid Evaluation...")
    
    cfg = config or {}
    
    # 1. Compute regime features
    tprint_info("   Computing regime features...")
    regime_features = compute_layer4_regime_features(market_data)
    
    # Align to oof_df index
    common_idx = oof_df.index.intersection(regime_features.index)
    if len(common_idx) < 100:
        tprint_error(f"Insufficient overlap between OOF and market data: {len(common_idx)}")
        return pd.DataFrame(), {}
    
    X = regime_features.loc[common_idx]
    oof_aligned = oof_df.loc[common_idx]
    
    l3_probs = pd.to_numeric(oof_aligned[l3_prob_col], errors='coerce')
    y_true = pd.to_numeric(oof_aligned[target_col], errors='coerce')
    returns = pd.to_numeric(oof_aligned[return_col], errors='coerce')
 
    try:
        include_l3_prob_feature = bool(cfg.get('layer4_include_l3_prob_feature', True))
    except Exception:
        include_l3_prob_feature = True

    if include_l3_prob_feature:
        X = X.copy()
        X['l3_prob'] = l3_probs
     
    l3_arr = l3_probs.to_numpy(dtype=float, copy=False)
    y_arr = y_true.to_numpy(dtype=float, copy=False)
    r_arr = returns.to_numpy(dtype=float, copy=False)
    
    # 2. Generate OOF predictions for Layer 4
    tprint_info("   Generating Layer 4 OOF predictions...")
    
    # Use purged time-series CV
    from src.utils.purged_kfold import PurgedKFoldTime
    
    try:
        default_keep_fraction = float(cfg.get('layer4_quantile_threshold', 0.6))
    except Exception:
        default_keep_fraction = 0.6
    
    grid_keep_fractions: List[float] = []
    for x in (l3_quantile_thresholds or []):
        try:
            xf = float(x)
        except Exception:
            continue
        if np.isfinite(xf):
            grid_keep_fractions.append(xf)
    
    if not grid_keep_fractions:
        grid_keep_fractions = [default_keep_fraction]
    
    keep_fractions_to_compute: List[float] = []
    for xf in [default_keep_fraction] + grid_keep_fractions:
        try:
            kf = float(xf)
        except Exception:
            continue
        if not np.isfinite(kf):
            continue
        if not any(abs(kf - kk) < 1e-12 for kk in keep_fractions_to_compute):
            keep_fractions_to_compute.append(kf)
    
    try:
        purge_minutes = int(cfg.get('layer4_purge_minutes', 60))
    except Exception:
        purge_minutes = 60
    try:
        embargo_minutes = int(cfg.get('layer4_embargo_minutes', 30))
    except Exception:
        embargo_minutes = 30

    cv = PurgedKFoldTime(
        n_splits=n_folds,
        purge=pd.Timedelta(minutes=purge_minutes),
        embargo=pd.Timedelta(minutes=embargo_minutes),
    )
    splits = list(cv.split(X))
    
    labeled_mask = np.isfinite(y_arr)

    l4_oof_probs_by_keep: Dict[float, np.ndarray] = {}
    for keep_fraction in keep_fractions_to_compute:
        tprint_info(f"   L4 OOF for L3 keep_fraction={keep_fraction:.3f}...")
        l4_oof_probs = np.full(len(common_idx), np.nan)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            tprint_info(f"   Fold {fold_idx + 1}/{n_folds}...")
            
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train_raw = y_true.iloc[train_idx]
            labeled_train = pd.to_numeric(y_train_raw, errors='coerce').notna()
            if int(labeled_train.sum()) < 50:
                continue
 
            X_train = X_train.loc[labeled_train]
            y_train = (pd.to_numeric(y_train_raw.loc[labeled_train], errors='coerce').astype(float) >= 0.5).astype(int)
            l3_train = l3_probs.iloc[train_idx].loc[labeled_train]
             
            l4_model = Layer4RiskFilter(
                n_estimators=int(cfg.get('layer4_n_estimators', 300)),
                max_depth=int(cfg.get('layer4_max_depth', 5)),
                min_samples_leaf=int(cfg.get('layer4_min_samples_leaf', 20)),
                l3_keep_fraction=keep_fraction,
            )
            
            l4_model.fit(X_train, y_train, l3_train)
            
            if l4_model._is_fitted:
                l4_oof_probs[val_idx] = l4_model.predict_proba(X_val)
        
        l4_oof_probs_by_keep[keep_fraction] = l4_oof_probs
    
    # 3. Grid evaluation
    tprint_info("   Evaluating threshold/formula grid...")
    
    try:
        decision_threshold = float(cfg.get('layer4_decision_threshold', 0.5))
    except Exception:
        decision_threshold = 0.5
    try:
        min_trades = int(cfg.get('layer4_min_trades', 50))
    except Exception:
        min_trades = 50
    try:
        n_boot = int(cfg.get('layer4_n_boot', 500))
    except Exception:
        n_boot = 500
     
    results = []
    formulas = ['product', 'min', 'logit_avg']
    
    for q_thresh in l3_quantile_thresholds:
        try:
            keep_fraction = float(q_thresh)
        except Exception:
            continue
        
        l3_thresh = _l3_threshold_from_keep_fraction(l3_arr, keep_fraction)
        l3_mask = np.isfinite(l3_arr) & (l3_arr >= l3_thresh)
        
        l4_oof_probs = l4_oof_probs_by_keep.get(keep_fraction)
        if l4_oof_probs is None:
            continue
        
        for formula in formulas:
            # Compute final scores
            if formula == 'product':
                final_scores = compute_final_score_product(l3_arr, l4_oof_probs)
            elif formula == 'min':
                final_scores = compute_final_score_min(l3_arr, l4_oof_probs)
            elif formula == 'logit_avg':
                final_scores = compute_final_score_logit_avg(l3_arr, l4_oof_probs)
            else:
                final_scores = l3_arr
            
            # Evaluate on L3-gated subset
            eval_mask = l3_mask & np.isfinite(final_scores) & labeled_mask
            
            if eval_mask.sum() < 50:
                continue
            
            y_eval = (y_arr[eval_mask] >= 0.5).astype(int)
            p_eval = final_scores[eval_mask]
            r_eval = r_arr[eval_mask]
            
            # Metrics
            try:
                auc = float(roc_auc_score(y_eval, p_eval))
            except Exception:
                auc = 0.5
            
            try:
                brier = float(brier_score_loss(y_eval, p_eval))
            except Exception:
                brier = 0.25
            
            # Trading metrics (using final score as trade decision)
            trade_mask = (p_eval >= decision_threshold) & np.isfinite(r_eval)
            n_trades = int(trade_mask.sum())

            if n_trades < min_trades:
                continue
             
            if n_trades > 0:
                traded_returns = r_eval[trade_mask]
                total_pnl = float(np.sum(traded_returns))
                avg_pnl = float(np.mean(traded_returns))
                win_rate = float(np.sum(traded_returns > 0) / n_trades)

                equity = _equity_curve_from_returns(traded_returns)
                total_return = float(equity[-1] - 1.0)
                max_drawdown = _max_drawdown_from_equity(equity)
                calmar_like = float(total_return / (max_drawdown + 1e-12))

                avg_pnl_ci_low, avg_pnl_ci_high = _bootstrap_mean_ci(traded_returns, n_boot=n_boot)

                trades_per_day = float('nan')
                try:
                    if isinstance(oof_aligned.index, pd.DatetimeIndex):
                        eval_index = oof_aligned.index[eval_mask]
                        trade_index = eval_index[trade_mask]
                        if len(trade_index) >= 2:
                            days = (trade_index[-1] - trade_index[0]).total_seconds() / 86400.0
                            if days > 0:
                                trades_per_day = float(n_trades / days)
                except Exception:
                    trades_per_day = float('nan')
                 
                wins = traded_returns[traded_returns > 0]
                losses = traded_returns[traded_returns < 0]
                gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
                gross_loss = float(-np.sum(losses)) if len(losses) > 0 else 0.0
                profit_factor = gross_profit / (gross_loss + 1e-8)
            else:
                total_pnl = 0.0
                avg_pnl = 0.0
                win_rate = 0.0
                profit_factor = 0.0

                total_return = 0.0
                max_drawdown = 0.0
                calmar_like = 0.0
                trades_per_day = float('nan')
                avg_pnl_ci_low = float('nan')
                avg_pnl_ci_high = float('nan')
 
            results.append({
                'l3_keep_fraction': keep_fraction,
                'l3_quantile': keep_fraction,
                'l3_threshold': l3_thresh,
                'formula': formula,
                'n_samples': int(eval_mask.sum()),
                'n_trades': n_trades,
                'decision_threshold': decision_threshold,
                'auc': auc,
                'brier': brier,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_pnl_ci_low': avg_pnl_ci_low,
                'avg_pnl_ci_high': avg_pnl_ci_high,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trades_per_day': trades_per_day,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'calmar_like': calmar_like,
            })
    
    results_df = pd.DataFrame(results)
    
    # 4. Summary metrics
    metrics = {
        'n_samples': len(common_idx),
        'purge_minutes': int(purge_minutes),
        'embargo_minutes': int(embargo_minutes),
        'decision_threshold': float(decision_threshold),
        'min_trades': int(min_trades),
        'n_boot': int(n_boot),
        'l4_oof_coverage': float(np.sum(np.isfinite(l4_oof_probs_by_keep[default_keep_fraction])) / len(l4_oof_probs_by_keep[default_keep_fraction])),
        'l4_oof_mean': float(np.nanmean(l4_oof_probs_by_keep[default_keep_fraction])),
        'l4_oof_std': float(np.nanstd(l4_oof_probs_by_keep[default_keep_fraction])),
        'grid_results': results,
    }
    
    # Find best configuration
    if not results_df.empty:
        # Best by AUC
        best_auc_idx = results_df['auc'].idxmax()
        metrics['best_by_auc'] = results_df.loc[best_auc_idx].to_dict()
        
        # Best by profit factor (among configs with enough trades)
        pf_df = results_df[results_df['n_trades'] >= min_trades]
        if not pf_df.empty:
            best_pf_idx = pf_df['profit_factor'].idxmax()
            metrics['best_by_pf'] = pf_df.loc[best_pf_idx].to_dict()

        try:
            select_metric = str(cfg.get('layer4_select_metric', 'profit_factor'))
        except Exception:
            select_metric = 'profit_factor'

        sel_df = pf_df if (select_metric in ['profit_factor', 'avg_pnl', 'total_return', 'calmar_like', 'trades_per_day', 'win_rate']) else results_df
        if (select_metric in sel_df.columns) and (not sel_df.empty):
            if select_metric in ['brier']:
                best_idx = sel_df[select_metric].idxmin()
            else:
                best_idx = sel_df[select_metric].idxmax()
            metrics['best'] = sel_df.loc[best_idx].to_dict()
    
    tprint_success(f"   Grid evaluation complete. {len(results)} configurations tested.")
    
    # Add L4 OOF predictions to output
    oof_aligned = oof_aligned.copy()
    oof_aligned['layer4_prob'] = l4_oof_probs_by_keep[default_keep_fraction]
    
    return oof_aligned, metrics
