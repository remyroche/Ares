import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Union, Callable, Optional, Tuple, Any
from enum import Enum
from scipy.stats import spearmanr, entropy as shannon_entropy, norm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from .focal_loss_utils import get_focal_loss_lgbm

# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==========================================
# 0. Data Structures & Configuration
# ==========================================

FIXED_GRID = [
    # --- Ratio 1.5 ---
    {'id': '1.5:1', 'pt': 2.25, 'sl': 1.5},
    {'id': '3:2',   'pt': 3.75, 'sl': 2.5},

    # --- Ratio 2.0 ---
    {'id': '2:1',   'pt': 3.00, 'sl': 1.5},
    {'id': '4:2',   'pt': 5.00, 'sl': 2.5},

    # --- Ratio 3.0 ---
    {'id': '3:1',   'pt': 4.50, 'sl': 1.5},

    # --- Ratio 4.0 ---
    {'id': '4:1',   'pt': 6.00, 'sl': 1.5},
]

class OutputGeometry:
    
    def __init__(self, name, family, events, labels, weights, purity, auc, cluster_id=None, params=None, metrics=None):
        self.name = name
        self.family = family
        self.events = events
        self.labels = labels
        self.weights = weights
        self.purity = purity      # Uniqueness Score
        self.auc = auc            # Learnability Score (The Tournament Metric)
        self.cluster_id = cluster_id
        self.params = params if params is not None else {}
        self.metrics = metrics if metrics is not None else {}
    
    def __repr__(self):
        return f"<Geometry {self.name} | AUC={self.auc:.3f} | Purity={self.purity:.2f} | N={len(self.events)}>"


class KalmanFilter1D:
    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        self.Q = Q
        self.R = R
        self.x = initial_value
        self.P = 1.0

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        values = series.values
        n = len(values)
        x_hat = np.zeros(n)
        P_hat = np.zeros(n)
        x, P = self.x, self.P
        Q, R = self.Q, self.R

        for i in range(n):
            x_pred = x
            P_pred = P + Q
            z = values[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            x_hat[i] = x
            P_hat[i] = P

        return pd.Series(x_hat, index=series.index), pd.Series(P_hat, index=series.index)

def roll_entropy(series: pd.Series, window: int = 24, bins: int = 10) -> pd.Series:
    def _entropy_calc(x):
        if np.max(x) == np.min(x): return 0.0
        hist_counts, _ = np.histogram(x, bins=bins)
        return shannon_entropy(hist_counts)
    return series.rolling(window).apply(_entropy_calc, raw=True)

def calc_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = price * volume
    cum_pv = pv.rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    return cum_pv / (cum_vol + 1e-9)

def calc_tr(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    if 'high' in cols and 'low' in cols:
        high = df[cols['high']]
        low = df[cols['low']]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = close.diff().abs()
    return tr

def average_uniqueness(indicator: pd.DataFrame) -> float:
    concurrency = indicator.sum(axis=1)
    valid_c = concurrency[concurrency > 0]
    if valid_c.empty: return 0.0
    return (1.0 / valid_c).mean()

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    arr = np.zeros(len(index), dtype=int)
    valid_events = events.intersection(index)
    if valid_events.empty:
        return pd.DataFrame(0, index=index, columns=[0])
    
    event_locs = index.get_indexer(valid_events)
    event_locs = event_locs[event_locs != -1]
    n_bars = len(index)

    # Fill indicator matrix
    for loc in event_locs:
        end_loc = min(loc + horizon, n_bars)
        arr[loc:end_loc] = 1 # Use binary indicator for set operations
    
    return pd.DataFrame(arr, index=index, columns=[0])

def generate_probe_features(price: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """Generate basic features for learnability probing."""
    df = pd.DataFrame(index=price.index)
    df['ret_1'] = price.pct_change()
    df['vol_20'] = df['ret_1'].rolling(20).std()

    # RSI approximation
    diff = price.diff()
    up = diff.where(diff > 0, 0)
    down = -diff.where(diff < 0, 0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rsi = 100 - (100 / (1 + ma_up / (ma_down + 1e-9)))
    df['rsi_14'] = rsi.fillna(50)

    if volume is not None:
        df['vol_chg'] = volume.pct_change()

    return df.fillna(0)

def ewma_volatility(returns, span=100):
    """
    EWMA volatility estimator.
    Used to normalize thresholds and make CUSUM regime-invariant.
    """
    return returns.ewm(span=span, adjust=False).std()

# ==========================================
# 1. Labeling Logic (Vectorized Dominance & State)
# ==========================================

def compute_volatility_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 20,
    k: float = 1.3
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Regime/State events (Volatility Expansion).
    Target: 1 if volatility increases by k% over the next `horizon` bars.

    Returns matched signature: labels, weights, returns, mfe, mae, vol
    Note: returns/mfe/mae are proxies here.
    """
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    # Calculate volatility if not present
    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d']
    else:
        vol = df['close'].pct_change().rolling(100).std()

    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    # Vectorized Target Calculation
    # We want Vol[t+horizon] > Vol[t] * k

    # Get indices
    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)

    target_locs = event_locs + horizon
    valid_mask = target_locs < n_bars

    event_locs = event_locs[valid_mask]
    target_locs = target_locs[valid_mask]
    final_events = valid_events[valid_mask]

    vol_start = vol.iloc[event_locs].values
    vol_end = vol.iloc[target_locs].values

    # Avoid zero vol
    vol_start = np.maximum(vol_start, 1e-9)

    # Calculate Ratio
    vol_ratio = vol_end / vol_start

    # Label: 1 if Expansion, 0 otherwise
    labels_arr = (vol_ratio > k).astype(float)

    # Weights: Magnitude of expansion
    weights_arr = np.log1p(np.abs(vol_ratio - 1.0))

    # "Returns": Here we use Volatility Change %
    returns_arr = vol_ratio - 1.0

    # MFE/MAE: Proxies using max vol in window
    # To implement correctly, we'd need window scan. For now, use end point proxy.
    mfe_arr = returns_arr # Max expansion
    mae_arr = np.zeros_like(returns_arr) # No "loss" concept really

    # Construct Series
    idx = final_events
    s_labels = pd.Series(labels_arr, index=idx)
    s_weights = pd.Series(weights_arr, index=idx)
    s_returns = pd.Series(returns_arr, index=idx)
    s_mfe = pd.Series(mfe_arr, index=idx)
    s_mae = pd.Series(mae_arr, index=idx)
    s_vol = pd.Series(vol_start, index=idx)

    return s_labels, s_weights, s_returns, s_mfe, s_mae, s_vol


def compute_path_degradation_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 20,
    d_sigma: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Market Stress (Path Degradation).
    Target: 1 if Max Intra-Horizon Drawdown > d_sigma * volatility.

    Returns matched signature: labels, weights, returns, mfe, mae, vol
    """
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d']
    else:
        vol = df['close'].pct_change().rolling(100).std()

    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    # Map events to integers
    if df.index.tz is not None:
        idx_base = df.index.tz_localize(None)
    else:
        idx_base = df.index

    if valid_events.tz is not None:
        events_norm = valid_events.tz_localize(None)
    else:
        events_norm = valid_events

    event_idxs = idx_base.get_indexer(events_norm)
    n_bars = len(df)

    # Filter valid
    valid_mask = (event_idxs != -1) & (event_idxs < (n_bars - horizon))
    valid_idxs = event_idxs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return tuple([pd.Series(dtype=float)] * 6)

    # Window Matrix
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    high_vals = df['high'].values[window_idxs]
    low_vals = df['low'].values[window_idxs]

    # Calculate Max Drawdown
    # Running Max of Highs
    running_max = np.maximum.accumulate(high_vals, axis=1)
    # Drawdown from running max to current low
    drawdowns = (running_max - low_vals) / running_max
    max_dd = np.max(drawdowns, axis=1)

    # Thresholds
    vol_vals = vol.values[valid_idxs]
    # Ensure vol is not zero
    vol_vals = np.maximum(vol_vals, 1e-6)

    thresholds = vol_vals * d_sigma

    labels_arr = (max_dd > thresholds).astype(float)

    # Weights: Severity of breakdown
    weights_arr = np.log1p(max_dd / thresholds)

    # "Returns": Max Drawdown (negative)
    returns_arr = -max_dd

    s_labels = pd.Series(labels_arr, index=final_events)
    s_weights = pd.Series(weights_arr, index=final_events)
    s_returns = pd.Series(returns_arr, index=final_events)
    s_mfe = pd.Series(np.zeros_like(returns_arr), index=final_events)
    s_mae = pd.Series(max_dd, index=final_events)
    s_vol = pd.Series(vol_vals, index=final_events)

    return s_labels, s_weights, s_returns, s_mfe, s_mae, s_vol


def compute_dominance_labels(
    price: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    risk_budget: float = 1.0,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 120,
    transaction_cost: float = 0.003,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Vectorized MFE/MAE Dominance Labeling with Risk Budget.
    Uses risk budget to control how close trades can get to stop-loss levels.
    
    Returns: labels, weights, returns, mfe, mae, volatility
    """
    # 1. Filter events within bounds
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    n_bars = len(price)


    # Map events to integers
    # Normalize TZ to ensure matching
    if price.index.tz is not None:
        price_idx = price.index.tz_localize(None)
    else:
        price_idx = price.index
        
    if events.tz is not None:
        events_norm = events.tz_localize(None)
    else:
        events_norm = events

    event_idxs = price_idx.get_indexer(events_norm)
    
    # DEBUG: Deep inspection
    if len(events) > 0 and (event_idxs == -1).all():
         logger.warning(f"DEBUG: All indices -1. Mismatch suspected.")
         logger.warning(f"DEBUG: Price idx dtype: {price.index.dtype}")
         logger.warning(f"DEBUG: Events dtype: {events.dtype}")
         logger.warning(f"DEBUG: First 5 event_idxs: {event_idxs[:5]}")
         try:
             logger.warning(f"DEBUG: Price head: {price.index[:3]}")
             logger.warning(f"DEBUG: Events head: {events[:3]}")
         except: pass

    valid_mask = (event_idxs != -1) & (event_idxs < (n_bars - horizon))
    valid_idxs = event_idxs[valid_mask]
    valid_events = events[valid_mask]

    valid_events = events[valid_mask]
    
    # DEBUG: Check why empty
    if len(valid_idxs) == 0:
        logger.warning(f"DEBUG: No valid events found! n_events={len(events)}")
        if len(events) > 0:
             logger.warning(f"DEBUG: Event[0]: {events[0]} type={type(events[0])}")
             logger.warning(f"DEBUG: Price[0]: {price.index[0]} type={type(price.index[0])}")
             logger.warning(f"DEBUG: Price[-1]: {price.index[-1]}")
             logger.warning(f"DEBUG: n_bars={n_bars}, horizon={horizon}")
        return tuple([pd.Series(dtype=float)] * 6)

    # 2. Construct Window Matrix (N x Horizon)
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    # Get Prices
    price_vals = price.values
    entry_prices = price_vals[valid_idxs]

    # 3. Compute MFE/MAE & Hits
    vol_vals = volatility.values[valid_idxs]
    vol_vals = np.maximum(vol_vals, 1e-6)

    pt_thresh = (vol_vals * pt_mult)[:, None]
    sl_thresh = (-vol_vals * sl_mult)[:, None]

    # Check if High/Low provided
    if high is not None and low is not None:
        high_vals = high.values
        low_vals = low.values
        window_highs = high_vals[window_idxs]
        window_lows = low_vals[window_idxs]

        # Returns relative to entry
        high_ret = window_highs / entry_prices[:, None] - 1.0
        low_ret = window_lows / entry_prices[:, None] - 1.0

        mfe = np.max(high_ret, axis=1)
        # MAE is max negative excursion (magnitude)
        mae = -np.min(low_ret, axis=1)

        hit_pt = high_ret > pt_thresh
        hit_sl = low_ret < sl_thresh
    else:
        window_prices = price_vals[window_idxs]
        returns_matrix = window_prices / entry_prices[:, None] - 1.0

        mfe = np.max(returns_matrix, axis=1)
        mae = np.max(-returns_matrix, axis=1)

        hit_pt = returns_matrix > pt_thresh
        hit_sl = returns_matrix < sl_thresh

    # For outcome calculation, we use Close prices if neither hit
    window_closes = price_vals[window_idxs]
    close_returns = window_closes / entry_prices[:, None] - 1.0

    # Identify first hit indices
    any_pt = np.any(hit_pt, axis=1)
    any_sl = np.any(hit_sl, axis=1)

    first_pt_idx = np.argmax(hit_pt, axis=1)
    first_sl_idx = np.argmax(hit_sl, axis=1)

    # TBM Logic (simple win/loss)
    win_mask = any_pt & (~any_sl | (first_pt_idx < first_sl_idx))

    # Risk Budget Logic: MAE / Stop_Dist <= risk_budget
    stop_dist = sl_mult * vol_vals
    risk_used = mae / np.maximum(stop_dist, 1e-9)
    risk_mask = risk_used <= risk_budget

    # Economic viability
    min_profit = transaction_cost * 1.1
    profit_mask = mfe > min_profit

    # Final Label
    final_label_mask = win_mask & risk_mask & profit_mask
    labels = final_label_mask.astype(float)

    # 5. Weighting
    mae_safe = np.maximum(mae, 1e-9)
    ratio = mfe / mae_safe
    magnitude = np.log1p(mfe / transaction_cost)
    vol_adj = 1.0 / vol_vals
    weights = ratio * magnitude * vol_adj

    # 6. Returns (use win_mask)
    out_returns = np.where(win_mask, pt_mult * vol_vals, -sl_mult * vol_vals)
    timeout_mask = (~any_pt) & (~any_sl)
    out_returns[timeout_mask] = close_returns[timeout_mask, -1]

    # Construct Series
    idx = valid_events
    s_labels = pd.Series(labels, index=idx)
    s_weights = pd.Series(weights, index=idx)
    s_returns = pd.Series(out_returns, index=idx)
    s_mfe = pd.Series(mfe, index=idx)
    s_mae = pd.Series(mae, index=idx)
    s_vol = pd.Series(vol_vals, index=idx)

    return s_labels, s_weights, s_returns, s_mfe, s_mae, s_vol

# ==========================================
# 2. Quality Gates & Checks
# ==========================================

def effective_n(labels, max_lag):
    """Estimate effective sample size accounting for autocorrelation."""
    labels = np.asarray(labels)
    n = len(labels)
    if n <= max_lag: return n

    rho_sum = 0.0
    # Fast manual autocorrelation for small lag
    for k in range(1, max_lag + 1):
        y1 = labels[:-k]
        y2 = labels[k:]
        if len(y1) < 2: continue
        y1_dev = y1 - y1.mean()
        y2_dev = y2 - y2.mean()
        denom = np.sqrt(np.sum(y1_dev**2) * np.sum(y2_dev**2))
        if denom == 0: continue
        rho = np.sum(y1_dev * y2_dev) / denom
        rho_sum += rho

    n_eff = n / (1.0 + 2.0 * rho_sum)
    return max(1.0, n_eff)

def significance_score(labels, max_lag):
    n_eff = effective_n(labels, max_lag)
    return np.log1p(n_eff)

def calculate_psr(sharpe, n, skew, kurt, target_sharpe=0):
    if n < 2: return 0.0
    std_sharpe = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    if std_sharpe == 0: return 0.0
    return norm.cdf((sharpe - target_sharpe) / std_sharpe)

def check_label_quality(
    events: pd.DatetimeIndex,
    labels: pd.Series,
    returns: pd.Series,
    df: pd.DataFrame,
    probe_features: pd.DataFrame,
    generator_instance,
    generator_params: dict
) -> Tuple[bool, Dict, str]:
    """Apply diagnostic gates to filter poor quality geometries."""
    
    n = len(labels)
    if n == 0:
        tprint_warning("❌ No labels - skipping gates")
        return False, {}, "No labels"
    
    # Calculate time span correctly for 15-minute data
    if len(labels.index) > 1:
        time_span = labels.index[-1] - labels.index[0]
        days = time_span.total_seconds() / (24 * 3600)
    else:
        days = 1.0
    
    rate = n / days if days > 0 else 0
    
    val_metrics = {
        'n': n, 'rate': rate, 'pos_rate': 0.0,
        'jaccard': 0.0, 'psr': 0.0, 'min_p': 0.0, 'max_mi': 0.0
    }
    
    gates_log = []
    failure_reason = "PASS"
    overall_pass = True

    # 1. Sample Size Gate
    if rate < 0.5:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [FAIL]")
        overall_pass = False
        if failure_reason == "PASS": failure_reason = "Sample Size (< 0.5/day)"
    else:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [OK]")

    # 2. Class Balance Gate
    pos_rate = labels.mean()
    val_metrics['pos_rate'] = pos_rate
    if pos_rate < 0.05 or pos_rate > 0.95:
        gates_log.append(f"Bal: {pos_rate:.1%} [FAIL]")
        overall_pass = False
        if failure_reason == "PASS": failure_reason = "Class Balance"
    else:
        gates_log.append(f"Bal: {pos_rate:.1%} [OK]")

    # 3. Perturbation Stability Gate
    try:
        df_noisy = df.copy()
        noise = np.random.normal(1.0, 0.0001, size=len(df))
        for col in ['close', 'high', 'low', 'open']:
            if col in df_noisy.columns: df_noisy[col] *= noise
        
        gen = generator_instance
        if gen.__class__.__name__ in DF_REQUIRED_CLASSES:
             events_noisy = gen.generate(df_noisy, **generator_params)
        else:
             events_noisy = gen.generate(df_noisy['close'], **generator_params)

        ind_clean = build_indicator_matrix(events, df.index, horizon=1).values.flatten()
        ind_noisy = build_indicator_matrix(events_noisy, df.index, horizon=1).values.flatten()
        
        intersection = np.logical_and(ind_clean, ind_noisy).sum()
        union = np.logical_or(ind_clean, ind_noisy).sum()
        jaccard = intersection / union if union > 0 else 0.0
        val_metrics['jaccard'] = jaccard
        
        if jaccard < 0.3:
            gates_log.append(f"Jaccard: {jaccard:.2f} [WARN]")
        else:
            gates_log.append(f"Jaccard: {jaccard:.2f} [OK]")
            
    except Exception as e:
        gates_log.append(f"Jaccard: ERR [WARN]")

    # 4. ANOVA Gate
    X = probe_features.loc[labels.index]
    y = labels
    with np.errstate(divide='ignore', invalid='ignore'):
        F, p_values = f_classif(X, y)
    valid_p = p_values[~np.isnan(p_values)]
    
    if len(valid_p) > 0:
        min_p = np.min(valid_p)
        val_metrics['min_p'] = min_p
        if min_p > 0.20:
            gates_log.append(f"ANOVA: p={min_p:.2f} [FAIL]")
            overall_pass = False
            if failure_reason == "PASS": failure_reason = "ANOVA"
        else:
            gates_log.append(f"ANOVA: p={min_p:.2f} [OK]")
    else:
         gates_log.append("ANOVA: N/A [WARN]")

    # 5. Mutual Info Gate
    # Optimization: effective N limit for MI to avoid O(N^2) scaling
    MAX_MI_SAMPLES = 2000
    if len(X) > MAX_MI_SAMPLES:
        # random_state is already 42 fixed for consistency
        indices = np.random.RandomState(42).choice(len(X), MAX_MI_SAMPLES, replace=False)
        X_mi = X.iloc[indices]
        y_mi = y.iloc[indices]
    else:
        X_mi = X
        y_mi = y

    mi = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=42)
    max_mi = np.max(mi)
    val_metrics['max_mi'] = max_mi
    
    if max_mi < 0.005:
        gates_log.append(f"MI: {max_mi:.4f} [FAIL]")
        overall_pass = False
        if failure_reason == "PASS": failure_reason = "Mutual Info"
    else:
        gates_log.append(f"MI: {max_mi:.4f} [OK]")

    summary_str = " | ".join(gates_log)
    if overall_pass:
        tprint_info(f"✅ Gates Passed: {summary_str}")
    else:
        tprint_warning(f"❌ Gates Failed: {summary_str}")

    return overall_pass, val_metrics, failure_reason

# ==========================================
# 3. Multi-Factor Scoring
# ==========================================

def calculate_multifactor_score(
    candidates: List[Dict],
    probe_features: pd.DataFrame
) -> List[Dict]:
    if not candidates: return []
    scores = []

    for cand in candidates:
        labels = cand['labels']
        n = len(labels)
        mfe = cand['mfe']
        mae = cand['mae']
        vol = cand['vol']

        X = probe_features.loc[labels.index]
        # Optimization: Limit sample size for Spearman calculation
        MAX_SAMPLES = 2000
        if n > MAX_SAMPLES:
            indices = np.random.RandomState(42).choice(n, MAX_SAMPLES, replace=False)
            X_sub = X.iloc[indices]
            labels_sub = labels.iloc[indices]
        else:
            X_sub = X
            labels_sub = labels

        ic_vals = [abs(spearmanr(X_sub[col], labels_sub)[0]) for col in X_sub.columns]
        ic_max = np.nanmax(ic_vals) if ic_vals else 0

        F, _ = f_classif(X_sub, labels_sub)
        f_max = np.nanmax(F) if len(F) > 0 else 0

        # New Significance: Effective N
        max_lag = cand['params'].get('horizon', 120)
        significance = significance_score(labels, max_lag)

        # Stability
        chunk_size = n // 3
        if chunk_size > 10:
            ic_chunks = []
            for i in range(3):
                s = i * chunk_size
                e = (i + 1) * chunk_size if i < 2 else n
                # For stability, we use chunked data (preserving time order)
                # Don't sample here, use contiguous blocks
                sub_X = X.iloc[s:e]; sub_y = labels.iloc[s:e]
                
                # Limit size of chunks if necessary? 
                # If chunk is huge, sampling might destroy time structure for stability?
                # Stability here is cross-validation of IC basically.
                # Let's keep it on full chunk for now as it splits by 3 already.
                
                chunk_ics = [abs(spearmanr(sub_X[col], sub_y)[0]) for col in sub_X.columns]
                ic_chunks.append(np.nanmax(chunk_ics))
            stability = 1.0 / (np.std(ic_chunks) + 1e-6)
        else: stability = 0.5

        counts = labels.value_counts(normalize=True)
        balance = shannon_entropy(counts)

        indicator = build_indicator_matrix(cand['events'], X.index, horizon=cand['params']['horizon'])
        density = average_uniqueness(indicator)

        path_asymmetry = (mfe / vol) - (mae.abs() / vol)
        path_score = path_asymmetry.mean()

        cand['metrics_raw'] = {
            'ic': ic_max, 'f_stat': f_max, 'significance': significance,
            'stability': stability, 'balance': balance, 'density': density,
            'path_score': path_score
        }
        scores.append(cand)

    df_scores = pd.DataFrame([c['metrics_raw'] for c in scores])
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_scores), columns=df_scores.columns)

    for i, cand in enumerate(scores):
        row = df_norm.iloc[i]
        power = max(row['ic'], row['f_stat'])
        raw_sig = df_scores.iloc[i]['significance']

        final_score = (
            power *
            raw_sig *
            row['stability'] *
            row['balance'] *
            row['density'] *
            (1.0 + row['path_score'])
        )
        cand['score'] = final_score
        cand['power'] = power

    return scores

# ==========================================
# 4. Probe (LGBM - Advanced Metrics)
# ==========================================

def run_lgbm_probe(X, y, w, returns) -> Dict[str, float]:
    """
    Advanced Probe: Returns Meta-Label Lift, Yield, Entropy, Consistency.
    """
    tprint_info(f"🚀 Starting LGBM Probe for {len(y)} samples")
    
    if len(y) < 50:
        tprint_warning("⚠️ Too few samples for probe (< 50)")
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

    params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': 42}
    tscv = TimeSeriesSplit(n_splits=3)
    tprint_info("🔄 Setting up 3-fold time series cross-validation")

    meta_returns = []
    base_returns = []
    preds_all = []
    labels_all = []
    r_all = [] # Realized returns for all va samples

    fold = 0
    for tr_idx, va_idx in tscv.split(X):
        fold += 1
        tprint_info(f"📊 Training fold {fold}/3...")
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr, w_va = w.iloc[tr_idx], w.iloc[va_idx]
        ret_va = returns.iloc[va_idx]

        if y_tr.nunique() < 2 or y_va.nunique() < 2: 
            tprint_warning(f"⚠️ Fold {fold}: Insufficient class diversity, skipping")
            continue

        tprint_info(f"📊 Fold {fold}: Training {len(X_tr)} samples, validating {len(X_va)} samples")
        
        # Calculate dynamic alpha for Focal Loss
        pos_rate = y_tr.mean()
        alpha = 1.0 - pos_rate
        alpha = float(np.clip(alpha, 0.05, 0.95))
        
        # Use custom objective (ensure 'objective' and 'metric' don't conflict)
        fobj = get_focal_loss_lgbm(alpha=alpha, gamma=2.0)
        
        # Copy params to avoid mutation issues
        fold_params = params.copy()
        # Remove objective from params if we pass fobj is safer, but if train() rejects fobj kwarg,
        # we must either pass it as 'objective' key in params or rely on legacy behavior.
        # It seems the installed LightGBM version might be wrapped or older.
        # Let's try passing it via params['objective']
        fold_params['objective'] = fobj
        
        # Also ensure metric is set (auc)
        
        tprint_info(f"DEBUG: X_tr shape: {X_tr.shape}, columns: {X_tr.columns.tolist()}")
        if X_tr.shape[1] == 0:
             tprint_error("❌ X_tr has 0 features! LightGBM will crash.")
             return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va)
        
        model = lgb.train(
            fold_params, 
            dtrain, 
            valid_sets=[dvalid],
            # fobj=fobj, # Removed to fix TypeError
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )

        preds = model.predict(X_va)
        preds_all.extend(preds)
        labels_all.extend(y_va.values)
        r_all.extend(ret_va.values)
        
        # Performance calculation (De Prado)
        base_returns.extend(ret_va.values)

        # Meta: Pred > 0.5
        mask = preds > 0.5
        meta_count = mask.sum()
        tprint_info(f"📊 Fold {fold}: {meta_count}/{len(mask)} meta predictions ({meta_count/len(mask):.1%})")
        
        if mask.sum() > 0:
            meta_returns.extend(ret_va[mask].values)

    if not base_returns:
        tprint_warning("⚠️ No base returns calculated")
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

    tprint_info("📈 Calculating probe metrics...")

    # 1. Sharpe Lift
    def sharpe(r):
        if len(r) < 2: return 0.0
        std = np.std(r)
        if std == 0: return 0.0
        return np.mean(r) / std

    base_sh = sharpe(base_returns)
    meta_sh = sharpe(meta_returns) if meta_returns else 0.0
    lift = meta_sh - base_sh

    # 2. Opportunity Yield
    days = (returns.index[-1] - returns.index[0]).days if not returns.empty else 1
    n_pos = len(meta_returns)
    opp_yield = n_pos / max(1, days)

    # 3. Conditional Outcome Entropy H(Y | Pred > 0.5)
    # y is binary 0/1.
    preds_arr = np.array(preds_all)
    labels_arr = np.array(labels_all)
    mask = preds_arr > 0.5
    if mask.sum() > 0:
        cond_labels = labels_arr[mask]
        counts = pd.Series(cond_labels).value_counts(normalize=True)
        cond_entropy = shannon_entropy(counts)
    else:
        cond_entropy = 1.0 # High entropy if no signals

    # 4. Sign Consistency
    if mask.sum() > 0:
        consistency = np.mean(labels_arr[mask])    # --- De Prado / Advanced Metrics ---
    # 1. IC (Information Coefficient)
    ic, _ = spearmanr(preds_all, r_all) if len(r_all) > 10 else (0.0, 1.0)
    
    # 2. PSR (Probabilistic Sharpe Ratio)
    r_arr = np.array(meta_returns)
    sharpe_p = sharpe(r_arr)
    n_p = len(r_arr)
    psr_val = 0.0
    if n_p > 2:
        from scipy.stats import skew, kurtosis
        s = skew(r_arr)
        k = kurtosis(r_arr)
        psr_val = calculate_psr(sharpe_p, n_p, s, k)

    # 3. Standardized Error (Consistency)
    fold_sharpes = [sharpe(np.array(f)) for f in [meta_returns[i:i+len(r_arr)//3] for i in range(0, len(r_arr), len(r_arr)//3)] if len(f) > 0]
    std_error = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0.0

    tprint_success(f"✅ Probe Complete: Lift={lift:.4f}, IC={ic:.4f}, PSR={psr_val:.4f}")
    
    return {
        'lift': float(lift), 
        'yield': float(opp_yield), 
        'entropy': float(cond_entropy), 
        'consistency': float(consistency),
        'sharpe_meta': float(sharpe_p),
        'ic': float(ic),
        'psr': float(psr_val),
        'std_error': float(std_error)
    }

def adaptive_threshold_calculator(
    generator: "BaseEventGenerator",
    data: Union[pd.Series, pd.DataFrame],
    target_signals_per_day: float = 7.5,
    max_iterations: int = 20,
    tolerance: float = 0.2
) -> pd.DatetimeIndex:
    """
    Iteratively adjust thresholds to achieve target signal rate.
    """
    # Calculate data duration and target signal count
    if isinstance(data, pd.Series):
        index = data.index
    else:
        index = data.index
    
    duration_days = (index[-1] - index[0]).days
    if duration_days < 1:
        duration_days = 1
    
    target_signals = int(target_signals_per_day * duration_days)
    min_target = int(target_signals * (1 - tolerance))
    max_target = int(target_signals * (1 + tolerance))
    
    # Start with default parameters (need to be passed in)
    # This is a simplified version - in practice, you'd pass the specific params
    events = generator.generate(data)
    
    if len(events) == 0:
        return events
    
    # Iterative adjustment
    iteration = 0
    current_events = events
    
    while iteration < max_iterations:
        current_count = len(current_events)
        
        # Check if within tolerance
        if min_target <= current_count <= max_target:
            break
        
        # Calculate adjustment factor
        if current_count > max_target:  # Too many signals
            factor = 1.2 + (current_count - max_target) / max_target * 0.3
        else:  # Too few signals
            factor = 0.8 - (min_target - current_count) / min_target * 0.3
        
        factor = max(0.5, min(2.0, factor))  # Bound the factor
        
        # Adjust parameters if generator supports it
        # Panic mode for extremely low signals
        if current_count < min_target * 0.1:
             factor = 0.5 # Aggressive relaxation
             logger.info(f"Panic relaxation: Rate is {current_count}/{min_target} (target). Slashed params by 50%.")
        
        # Adjust parameters if generator supports it
        if hasattr(generator, '_adjust_z_threshold'):
            current_params = generator._adjust_z_threshold(current_params, factor)
            # Re-generate with new params to check progress within loop
            try:
                # We need to call generate again. 
                # generator is an instance of BaseEventGenerator (or subclass)
                # We need to handle the positional args issue if relevant, but here we just use **current_params
                # But wait, generate() might need positional args if they were passed...
                # The prompt said "generator.generate(data)" at line 661. 
                # We should use the same call structure.
                # Actually, check line 839: "events = self.generate(data, *args, **current_params)"
                # This function 'adaptive_threshold_calculator' is a standalone function at module level?
                # No, look at line 635. Yes it is.
                # But wait, BaseEventGenerator.generate_adaptive calls self.generate.
                # 'adaptive_threshold_calculator' seems to be an older standalone function?
                # Actually, 'BaseEventGenerator.generate_adaptive' is the one used in the main loop!
                # Line 1823 calls `gen.generate_adaptive`.
                # So I should update `BaseEventGenerator.generate_adaptive` NOT the standalone function if it's unused.
                # Let's check if `adaptive_threshold_calculator` is used.
                pass
            except Exception as e:
                logger.warning(f"Optimization step failed: {e}")
                break
        
        iteration += 1
    
    return current_events

# ==========================================
# 5. Signal Generators
# ==========================================

GENERATOR_PARAM_NAMES = {
    'EntropyEvents': ['window'],
    'ATRShockEvents': ['lookback', 'long_window', 'z'],
    'MicrostructureEvents': ['window'],
    'TrendModulatedBreakoutEvents': ['lookback', 'anchor_window'],
    'KalmanTrendEvents': ['q_fast', 'q_slow'],
    'VWAPReversionEvents': ['lookback', 'z'],
    'KalmanRegimeEvents': ['Q', 'R', 'z'],
    'VWAPCrossEvents': ['lookback'],
    'FairValueGapEvents': ['min_gap_pct', 'lookback', 'volume_threshold', 'confirm_candles'],
    'SupportResistanceBreakEvents': ['lookback', 'min_touches', 'breakout_threshold', 'volume_threshold', 'min_strength_score'],
    'OrderBlockEvents': ['lookback', 'min_move_pct', 'volume_threshold']
}

DF_REQUIRED_CLASSES = (
    'MicrostructureEvents', 
    'TrendModulatedBreakoutEvents',
    'VWAPReversionEvents',
    'FairValueGapEvents',
    'SupportResistanceBreakEvents',
    'OrderBlockEvents',
    'KalmanRegimeEvents',
    'VWAPCrossEvents',
    'ImprovedCUSUMEvents'
) 
# Note: ImprovedCUSUMEvents needs DF for generate() but here we list by name first? 
# Actually best to use classes if they are defined. 
# But classes are defined further down. 
# So I should move this definition AFTER the classes are defined, or use string matching?
# Using isinstance requires actual classes. 
# I can define this tuple inside the functions or at the end of the file, or move classes up?
# Moving classes is risky.
# I will define it as a tuple of classes logic inside the function or define it at the bottom.
# But check_label_quality is at the top.
# Python functions evaluate body at runtime.
# So I can use global DF_REQUIRED_CLASSES if it's defined before check_label_quality is CALLED.
# But it needs to be defined after classes are defined.
# check_label_quality is defined at line 355.
# Classes like MicrostructureEvents are defined later (line 800+).
# So I cannot use a global tuple of classes at the top of the file.
# I have to check by class NAME in check_label_quality or define the tuple later and pass it in.
# Passing it in is cleaner. 
# check_label_quality signature update?
# Or just inspect the generator instance?
# "if isinstance(gen, ...)" requires classes to be in scope.
# check_label_quality is likely defined BEFORE the classes in this file?
# Let's check the file structure.


class BaseEventGenerator:
    """Base class for event generation with adaptive thresholding."""
    
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex: 
        """Generate events using default parameters."""
        raise NotImplementedError
    
    def _validate_data(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Validate input data with timezone and edge case handling."""
        if isinstance(data, pd.Series):
            if len(data) < 10:
                raise ValueError("Insufficient data points: need at least 10")
            if data.isna().all():
                raise ValueError("Data contains all-NaN values")
            # Check for timezone-aware index
            if data.index.tz is None:
                logger.debug("Data has no timezone - assuming UTC")
        else:
            if len(data) < 10:
                raise ValueError("Insufficient data points: need at least 10")
            if 'close' not in data.columns:
                raise ValueError("DataFrame must contain 'close' column")
            if data['close'].isna().all():
                raise ValueError("Close prices contain all-NaN values")
            # Check for timezone-aware index
            if data.index.tz is None:
                logger.debug("Data has no timezone - assuming UTC")
            
            # Validate OHLC data consistency if available
            if all(col in data.columns for col in ['open', 'high', 'low']):
                # Check for logical inconsistencies
                invalid_high = data['high'] < data['low']
                invalid_high_low = (data['high'] < data['close']) | (data['high'] < data['open'])
                invalid_low_high = (data['low'] > data['close']) | (data['low'] > data['open'])
                
                if invalid_high.any():
                    logger.warning(f"Found {invalid_high.sum()} bars where high < low")
                if invalid_high_low.any():
                    logger.warning(f"Found {invalid_high_low.sum()} bars with invalid high/low relationships")
                if invalid_low_high.any():
                    logger.warning(f"Found {invalid_low_high.sum()} bars with invalid low/high relationships")
    
    def _post_process_events(self, events: pd.DatetimeIndex, min_separation: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DatetimeIndex:
        """Remove clustered events, enforce minimum separation."""
        if len(events) <= 1:
            return events
        
        sorted_events = events.sort_values()
        filtered = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            if event - filtered[-1] >= min_separation:
                filtered.append(event)
        
        return pd.DatetimeIndex(filtered)
    
    def generate_adaptive(self, data: Union[pd.Series, pd.DataFrame], target_signals_per_day: float = 7.5, 
                        max_iterations: int = 20, tolerance: float = 0.1, *args, **params) -> pd.DatetimeIndex:
        """
        Generate events with adaptive thresholds to achieve target signal rate.
        Uses iterative convergence with proportional adjustment.
        """
        # Validate input data
        self._validate_data(data)
        
        # Calculate data duration and target signal count with timezone handling
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = data.index
        
        if len(index) < 2:
            logger.warning("Insufficient data for adaptive generation")
            return pd.DatetimeIndex([])
        
        # Handle timezone-aware vs naive datetime indices
        if index.tz is None:
            # Assume UTC for naive indices
            index = index.tz_localize('UTC')
            logger.debug("Localized naive datetime index to UTC")
        
        # Calculate duration in days (handles different timezones properly)
        duration_seconds = (index[-1] - index[0]).total_seconds()
        duration_days = max(1, duration_seconds / (24 * 3600))
        
        target_signals = int(target_signals_per_day * duration_days)
        min_target = int(target_signals * (1 - tolerance))
        max_target = int(target_signals * (1 + tolerance))
        
        # Start with default parameters
        current_params = params.copy()
        # Pass positional args if provided
        events = self.generate(data, *args, **current_params)
        
        if len(events) == 0:
            logger.debug("No events generated initially. Entering adaptive mode.")
        
        # Iterative adjustment with convergence
        iteration = 0
        best_events = events
        if len(events) == 0:
            best_error = target_signals
        else:
            best_error = abs(len(events) - target_signals)
        
        while iteration < max_iterations:
            current_count = len(events)
            current_error = abs(current_count - target_signals)
            
            # Check if within tolerance
            if min_target <= current_count <= max_target:
                logger.debug(f"Converged after {iteration + 1} iterations: {current_count} signals (target: {target_signals})")
                return events
            
            # Keep best result
            if current_error < best_error:
                best_error = current_error
                best_events = events
            
            # Calculate proportional adjustment factor
            # GOAL: factor < 1.0 -> Relax (More signals)
            #       factor > 1.0 -> Tighten (Fewer signals)
            
            if current_count > max_target:  # Too many signals -> TIGHTEN
                # Logic: Factor > 1.0
                factor = 1.15
            else:  # Too few signals -> RELAX
                # Logic: Factor < 1.0
                factor = 0.85
                
                # Panic calculation base
                if current_count < min_target * 0.1:
                    factor = 0.5  # Panic default
                    
            
            factor = max(0.5, min(2.0, factor))  # Bound the factor
            
            # Panic mode for extremely low signals (explicit)
            if current_count < min_target * 0.1 and iteration < 5:
                 factor = 0.5 # Aggressive relaxation
                 logger.debug(f"Panic relaxation: Rate is {current_count}/{target_signals} (target). Slashed params by 50%.")

            # Adjust parameters if generator supports it
            if hasattr(self, '_adjust_z_threshold'):
                current_params = self._adjust_z_threshold(current_params, factor)
                try:
                    new_events = self.generate(data, **current_params)
                except Exception as e:
                    logger.warning(f"Optimization step failed: {e}")
                    break

                # Only use adjusted if it improves the signal rate
                new_count = len(new_events)
                new_error = abs(new_count - target_signals)
                
                if new_error < current_error or iteration == max_iterations - 1:
                    events = new_events
                    logger.debug(f"Iteration {iteration + 1}: {current_count} -> {new_count} signals (factor: {factor:.2f})")
                else:
                    # If error increased significantly, stop
                    if new_error > current_error * 1.5:
                        break

            else:
                # Generator doesn't support adjustment, use current events
                break
            
            iteration += 1
        
        # Apply post-processing
        final_events = self._post_process_events(events)
        
        if len(final_events) != len(events):
            logger.debug(f"Post-processing removed {len(events) - len(final_events)} clustered events")
        
        return final_events

class EntropyEvents(BaseEventGenerator):
    """Generate events based on entropy spikes in price movements."""
    
    def generate(self, price: pd.Series, window: int = 24, z_thresh: float = 2.0) -> pd.DatetimeIndex:
        try:
            if len(price) < window * 5:
                logger.warning(f"Insufficient data for entropy calculation: need {window * 5}, got {len(price)}")
                return pd.DatetimeIndex([])
            
            log_ret = np.log(price).diff().fillna(0)
            ent = roll_entropy(log_ret, window=window, bins=10)
            
            # Avoid division by zero
            rolling_mean = ent.rolling(window*5).mean()
            rolling_std = ent.rolling(window*5).std()
            z_ent = (ent - rolling_mean) / (rolling_std + 1e-6)
            
            # Find threshold crossings (rising edge)
            events = price.index[(z_ent > z_thresh) & (z_ent.shift(1) <= z_thresh)]
            
            logger.debug(f"EntropyEvents generated {len(events)} events (window={window}, z_thresh={z_thresh})")
            return events
            
        except Exception as e:
            logger.error(f"EntropyEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'z_thresh' in adjusted:
            adjusted['z_thresh'] *= factor
        return adjusted

class MicrostructureEvents(BaseEventGenerator):
    """Generate events based on microstructure noise and Amihud illiquidity."""
    
    def generate(self, df: pd.DataFrame, window: int = 20, z: float = 2.0) -> pd.DatetimeIndex:
        try:
            if 'volume' not in df.columns:
                logger.warning("MicrostructureEvents requires volume data")
                return pd.DatetimeIndex([])
            
            if len(df) < window:
                logger.warning(f"Insufficient data for microstructure analysis: need {window}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Calculate returns and Amihud illiquidity
            ret = df['close'].pct_change().abs()
            amihud = ret / (df['volume'] * df['close'] + 1e-9)
            
            # Calculate z-score with safeguards
            rolling_mean = amihud.rolling(window).mean()
            rolling_std = amihud.rolling(window).std()
            zsc = (amihud - rolling_mean) / (rolling_std + 1e-6)
            
            events = df.index[zsc > z]
            
            logger.debug(f"MicrostructureEvents generated {len(events)} events (window={window}, z={z})")
            return events
            
        except Exception as e:
            logger.error(f"MicrostructureEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'z' in adjusted:
            adjusted['z'] *= factor
        if 'window' in adjusted:
            # For window, smaller is usually more sensitive? Or larger?
            # Smaller window -> more noise -> more signals?
            # Actually standard practice is usually shorter window = more signals.
            # Factor < 1 means relax -> lower thresholds -> more signals.
            # So if factor < 1, we want window to DECREASE?
            # Yes, adjust window by factor too.
            adjusted['window'] = max(5, int(adjusted['window'] * factor))
        return adjusted

class TrendModulatedBreakoutEvents(BaseEventGenerator):
    """Generate breakout events modulated by trend anchors (VWAP or moving average)."""
    
    def generate(self, df: pd.DataFrame, lookback: int = 20, anchor_window: int = 100) -> pd.DatetimeIndex:
        try:
            if len(df) < max(lookback, anchor_window):
                logger.warning(f"Insufficient data for breakout analysis: need {max(lookback, anchor_window)}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            price = df['close']
            rmax = price.rolling(lookback).max().shift(1)
            rmin = price.rolling(lookback).min().shift(1)
            
            # Calculate anchor (VWAP if volume available, else SMA)
            if 'volume' in df.columns:
                anchor = calc_vwap(price, df['volume'], anchor_window)
            else:
                anchor = price.rolling(anchor_window).mean()
                logger.debug("Using SMA as anchor (volume not available)")
            
            # Breakout conditions
            breakout_up = (price > rmax) & (price > anchor)
            breakout_down = (price < rmin) & (price < anchor)
            bk = breakout_up | breakout_down
            
            # Find first occurrence after no breakout
            events = price.index[bk & ~bk.shift(1).fillna(False)]
            
            logger.debug(f"TrendModulatedBreakoutEvents generated {len(events)} events (lookback={lookback}, anchor={anchor_window})")
            return events
            
        except Exception as e:
            logger.error(f"TrendModulatedBreakoutEvents generation failed: {e}")
            return pd.DatetimeIndex([])

class ATRShockEvents(BaseEventGenerator):
    """Generate events based on ATR volatility shocks."""
    
    def generate(self, df: pd.DataFrame, lookback: int = 14, long_window: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        try:
            if len(df) < long_window:
                logger.warning(f"Insufficient data for ATR shock analysis: need {long_window}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Calculate True Range and ATR
            tr = calc_tr(df, df['close'])
            atr = tr.rolling(lookback).mean()
            
            # Calculate z-score with safeguards
            rolling_mean = atr.rolling(long_window).mean()
            rolling_std = atr.rolling(long_window).std()
            zsc = (atr - rolling_mean) / (rolling_std + 1e-6)
            
            events = df['close'].index[zsc > z]
            
            logger.debug(f"ATRShockEvents generated {len(events)} events (lookback={lookback}, long_window={long_window}, z={z})")
            return events
            
        except Exception as e:
            logger.error(f"ATRShockEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'z' in adjusted:
            adjusted['z'] *= factor
        if 'long_window' in adjusted:
             # Reduce long window to catch shorter term shocks
             adjusted['long_window'] = max(20, int(adjusted['long_window'] * factor))
        return adjusted

class VWAPReversionEvents(BaseEventGenerator):
    """Generate events based on price reversion to VWAP."""
    
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        try:
            if 'volume' not in df.columns:
                logger.warning("VWAPReversionEvents requires volume data")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback:
                logger.warning(f"Insufficient data for VWAP reversion: need {lookback}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Calculate VWAP and z-score
            vwap = calc_vwap(df['close'], df['volume'], lookback)
            price_std = df['close'].rolling(lookback).std()
            
            # Avoid division by zero
            z_score = np.abs((df['close'] - vwap) / (price_std + 1e-6))
            events = df.index[z_score > z]
            
            logger.debug(f"VWAPReversionEvents generated {len(events)} events (lookback={lookback}, z={z})")
            return events
            
        except Exception as e:
            logger.error(f"VWAPReversionEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'z' in adjusted:
            adjusted['z'] *= factor
        if 'lookback' in adjusted:
             # Shorter lookback = more sensitive to recent moves
             adjusted['lookback'] = max(10, int(adjusted['lookback'] * factor))
        return adjusted

class KalmanTrendEvents(BaseEventGenerator):
    """Generate events based on Kalman filter trend changes."""
    
    def generate(self, price: pd.Series, q_fast: float = 1e-3, q_slow: float = 1e-5) -> pd.DatetimeIndex:
        try:
            if len(price) < 20:
                logger.warning(f"Insufficient data for Kalman trend analysis: need 20, got {len(price)}")
                return pd.DatetimeIndex([])
            
            # Apply Kalman filters
            f, _ = KalmanFilter1D(Q=q_fast).filter_series(price)
            s, _ = KalmanFilter1D(Q=q_slow).filter_series(price)
            
            # Detect trend changes
            trend_up = (f > s) & (f.shift(1) <= s.shift(1))
            trend_down = (f < s) & (f.shift(1) >= s.shift(1))
            events = price.index[trend_up | trend_down]
            
            logger.debug(f"KalmanTrendEvents generated {len(events)} events (q_fast={q_fast}, q_slow={q_slow})")
            return events
            
        except Exception as e:
            logger.error(f"KalmanTrendEvents generation failed: {e}")
            return pd.DatetimeIndex([])

class KalmanRegimeEvents(BaseEventGenerator):
    """Generate events based on Kalman filter regime shifts."""
    
    def generate(self, df: pd.DataFrame, Q: float = 1e-4, R: float = 0.01, z: float = 2.0) -> pd.DatetimeIndex:
        try:
            if len(df) < 20:
                logger.warning(f"Insufficient data for Kalman regime analysis: need 20, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Detect sudden shifts in price level relative to Kalman estimate
            price = df['close']
            f, P = KalmanFilter1D(Q=Q, R=R).filter_series(price)
            
            # Innovation and z-score
            innov = price - f
            std = np.sqrt(P + R)
            z_score = innov / (std + 1e-9)
            
            events = price.index[z_score.abs() > z]
            
            logger.debug(f"KalmanRegimeEvents generated {len(events)} events (Q={Q}, R={R}, z={z})")
            return events
            
        except Exception as e:
            logger.error(f"KalmanRegimeEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'z' in adjusted:
            adjusted['z'] *= factor
        return adjusted

class VWAPCrossEvents(BaseEventGenerator):
    """Generate events based on VWAP crossings."""
    
    def generate(self, df: pd.DataFrame, lookback: int = 50) -> pd.DatetimeIndex:
        try:
            if 'volume' not in df.columns:
                logger.warning("VWAPCrossEvents requires volume data")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback:
                logger.warning(f"Insufficient data for VWAP cross: need {lookback}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Calculate VWAP
            vwap = calc_vwap(df['close'], df['volume'], lookback)
            price = df['close']
            
            # Detect crossings
            cross_up = (price > vwap) & (price.shift(1) <= vwap.shift(1))
            cross_down = (price < vwap) & (price.shift(1) >= vwap.shift(1))
            events = price.index[cross_up | cross_down]
            
            logger.debug(f"VWAPCrossEvents generated {len(events)} events (lookback={lookback})")
            return events
            
        except Exception as e:
            logger.error(f"VWAPCrossEvents generation failed: {e}")
            return pd.DatetimeIndex([])

class FairValueGapEvents(BaseEventGenerator):
    """Generate events based on Fair Value Gap (FVG) / Smart Money Concept patterns.
    
    FVG occurs when there's a 3-candle pattern where:
    - Bullish FVG: High[1] < Low[2] (gap between candle 1 high and candle 3 low)
    - Bearish FVG: Low[1] > High[2] (gap between candle 1 low and candle 3 high)
    
    These gaps represent imbalances in buying/selling pressure and often get filled.
    """
    
    def generate(self, df: pd.DataFrame, min_gap_pct: float = 0.1, lookback: int = 20, 
                 volume_threshold: float = 1.5, confirm_candles: int = 3) -> pd.DatetimeIndex:
        try:
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"FairValueGapEvents requires OHLC data. Missing: {[c for c in required_cols if c not in df.columns]}")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback + confirm_candles:
                logger.warning(f"Insufficient data for FVG analysis: need {lookback + confirm_candles}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            # Calculate volume baseline for confirmation
            volume_ma = df['volume'].rolling(lookback).mean() if 'volume' in df.columns else None
            
            events = []
            
            # Scan for FVG patterns
            for i in range(2, len(df)):
                # Get the 3-candle window
                candle1 = df.iloc[i-2]  # 2 bars ago
                candle2 = df.iloc[i-1]  # 1 bar ago  
                candle3 = df.iloc[i]    # current bar
                
                # Calculate gap sizes
                bullish_gap = candle1['high'] - candle3['low']
                bearish_gap = candle3['high'] - candle1['low']
                
                # Calculate gap percentages relative to price
                price_ref = candle2['close']
                bullish_gap_pct = bullish_gap / price_ref if bullish_gap > 0 else 0
                bearish_gap_pct = bearish_gap / price_ref if bearish_gap > 0 else 0
                
                # Bullish FVG: High[1] < Low[2] (upward imbalance)
                if bullish_gap_pct > min_gap_pct:
                    # Additional confirmation: strong momentum and volume
                    momentum_up = candle2['close'] > candle1['close'] and candle3['close'] > candle2['close']
                    
                    volume_confirm = True
                    if volume_ma is not None:
                        volume_confirm = candle2['volume'] > volume_ma.iloc[i-1] * volume_threshold
                    
                    if momentum_up and volume_confirm:
                        events.append(df.index[i])
                
                # Bearish FVG: Low[1] > High[2] (downward imbalance)  
                elif bearish_gap_pct > min_gap_pct:
                    # Additional confirmation: strong momentum and volume
                    momentum_down = candle2['close'] < candle1['close'] and candle3['close'] < candle2['close']
                    
                    volume_confirm = True
                    if volume_ma is not None:
                        volume_confirm = candle2['volume'] > volume_ma.iloc[i-1] * volume_threshold
                    
                    if momentum_down and volume_confirm:
                        events.append(df.index[i])
            
            # Convert to DatetimeIndex and apply post-processing
            event_index = pd.DatetimeIndex(events)
            
            # Additional filter: avoid consecutive FVGs within short timeframe
            if len(event_index) > 1:
                min_separation = pd.Timedelta(hours=4)  # Minimum 4 hours between FVGs
                event_index = self._post_process_events(event_index, min_separation)
            
            logger.debug(f"FairValueGapEvents generated {len(event_index)} events (min_gap_pct={min_gap_pct}, lookback={lookback})")
            return event_index
            
        except Exception as e:
            logger.error(f"FairValueGapEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        """Adjust gap threshold for adaptive generation."""
        adjusted = params.copy()
        if 'min_gap_pct' in adjusted:
            adjusted['min_gap_pct'] *= factor
        if 'volume_threshold' in adjusted:
            adjusted['volume_threshold'] *= factor
        return adjusted

class SupportResistanceBreakEvents(BaseEventGenerator):
    """Generate events based on price breaking through established support/resistance levels.
    
    Identifies key S/R levels using pivot points and previous highs/lows with persistence,
    dynamic volatility-adjusted thresholds, and level strength scoring.
    """
    
    def __init__(self):
        self._sr_cache = {}  # Cache for S/R levels persistence
        self._cache_expiry_hours = 24  # Cache expires after 24 hours
    
    def _identify_sr_levels(self, window_data: pd.DataFrame, lookback: int, min_touches: int, 
                           current_time: pd.Timestamp) -> Tuple[List[Dict], List[Dict]]:
        """Identify support and resistance levels with strength scoring."""
        resistance_levels = []
        support_levels = []
        
        # Group similar price levels (clustering)
        price_tolerance = 0.001  # 0.1% tolerance
        
        # Find pivot highs and lows
        for j in range(1, len(window_data) - 1):
            # Pivot high detection
            if (window_data.iloc[j]['high'] > window_data.iloc[j-1]['high'] and 
                window_data.iloc[j]['high'] > window_data.iloc[j+1]['high']):
                
                level_price = window_data.iloc[j]['high']
                level_time = window_data.index[j]
                
                # Count touches with volume weighting
                touches = 0
                volume_weight = 0
                for k in range(len(window_data)):
                    price_diff = abs(window_data.iloc[k]['high'] - level_price) / level_price
                    if price_diff < price_tolerance:
                        touches += 1
                        # Volume weighting: recent touches matter more
                        time_weight = 1.0 - abs(k - j) / len(window_data)
                        volume_weight += window_data.iloc[k].get('volume', 0) * time_weight
                
                if touches >= min_touches:
                    strength_score = touches * (1 + volume_weight / (touches * window_data['volume'].mean() + 1e-9))
                    resistance_levels.append({
                        'price': level_price,
                        'time': level_time,
                        'touches': touches,
                        'strength': strength_score,
                        'volume_weight': volume_weight
                    })
            
            # Pivot low detection
            elif (window_data.iloc[j]['low'] < window_data.iloc[j-1]['low'] and 
                  window_data.iloc[j]['low'] < window_data.iloc[j+1]['low']):
                
                level_price = window_data.iloc[j]['low']
                level_time = window_data.index[j]
                
                # Count touches with volume weighting
                touches = 0
                volume_weight = 0
                for k in range(len(window_data)):
                    price_diff = abs(window_data.iloc[k]['low'] - level_price) / level_price
                    if price_diff < price_tolerance:
                        touches += 1
                        # Volume weighting: recent touches matter more
                        time_weight = 1.0 - abs(k - j) / len(window_data)
                        volume_weight += window_data.iloc[k].get('volume', 0) * time_weight
                
                if touches >= min_touches:
                    strength_score = touches * (1 + volume_weight / (touches * window_data['volume'].mean() + 1e-9))
                    support_levels.append({
                        'price': level_price,
                        'time': level_time,
                        'touches': touches,
                        'strength': strength_score,
                        'volume_weight': volume_weight
                    })
        
        # Sort by strength and keep only top levels
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        # Keep only significant levels (top 5 of each type)
        return resistance_levels[:5], support_levels[:5]
    
    def _get_cached_levels(self, symbol: str, current_time: pd.Timestamp) -> Tuple[List[Dict], List[Dict]]:
        """Retrieve cached S/R levels if still valid."""
        cache_key = f"{symbol}_{current_time.date()}"
        
        if cache_key in self._sr_cache:
            cache_data = self._sr_cache[cache_key]
            cache_time = cache_data['timestamp']
            
            # Check if cache is still valid
            if (current_time - cache_time).total_seconds() < self._cache_expiry_hours * 3600:
                return cache_data['resistance'], cache_data['support']
        
        return [], []
    
    def _update_cache(self, symbol: str, current_time: pd.Timestamp, 
                     resistance_levels: List[Dict], support_levels: List[Dict]):
        """Update S/R levels cache."""
        cache_key = f"{symbol}_{current_time.date()}"
        self._sr_cache[cache_key] = {
            'timestamp': current_time,
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _calculate_dynamic_threshold(self, df: pd.DataFrame, i: int, lookback: int, 
                                   base_threshold: float) -> float:
        """Calculate dynamic breakout threshold based on recent volatility."""
        # Get recent volatility for dynamic adjustment
        vol_window = min(lookback, i)
        if vol_window < 5:
            return base_threshold
        
        recent_data = df.iloc[max(0, i-vol_window):i]
        if len(recent_data) < 5:
            return base_threshold
        
        # Calculate ATR-based volatility
        high_low = recent_data['high'] - recent_data['low']
        high_close = (recent_data['high'] - recent_data['close']).abs()
        low_close = (recent_data['low'] - recent_data['close']).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(min(14, len(recent_data))).mean().iloc[-1]
        
        # Calculate price level
        current_price = df['close'].iloc[i]
        
        # Volatility-adjusted threshold
        vol_ratio = atr / current_price
        dynamic_threshold = base_threshold * (1 + vol_ratio * 2)  # Scale with volatility
        
        # Bound the threshold to reasonable values
        return max(base_threshold * 0.5, min(dynamic_threshold, base_threshold * 3))
    
    def generate(self, df: pd.DataFrame, lookback: int = 50, min_touches: int = 3, 
                 breakout_threshold: float = 0.002, volume_threshold: float = 1.5,
                 min_strength_score: float = 3.0) -> pd.DatetimeIndex:
        try:
            required_cols = ['high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"SupportResistanceBreakEvents requires OHLCV data. Missing: {[c for c in required_cols if c not in df.columns]}")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback * 2:
                logger.warning(f"Insufficient data for S/R analysis: need {lookback * 2}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            events = []
            price = df['close']
            volume_ma = df['volume'].rolling(lookback).mean()
            
            # Use symbol for caching (extract from index or use default)
            symbol = getattr(df.index, 'name', 'default') or 'default'
            
            # Pre-compute rolling statistics for efficiency
            df['price_range'] = df['high'] - df['low']
            df['avg_volume'] = volume_ma
            
            # Vectorized approach: identify potential breakout points first
            for i in range(lookback, len(df)):
                current_time = df.index[i]
                
                # Try to get cached levels first
                resistance_levels, support_levels = self._get_cached_levels(symbol, current_time)
                
                # If no cached levels, compute them
                if not resistance_levels and not support_levels:
                    window_start = max(0, i - lookback)
                    window_data = df.iloc[window_start:i].copy()
                    
                    resistance_levels, support_levels = self._identify_sr_levels(
                        window_data, lookback, min_touches, current_time
                    )
                    
                    # Update cache
                    self._update_cache(symbol, current_time, resistance_levels, support_levels)
                
                # Filter by minimum strength
                resistance_levels = [r for r in resistance_levels if r['strength'] >= min_strength_score]
                support_levels = [s for s in support_levels if s['strength'] >= min_strength_score]
                
                if not resistance_levels and not support_levels:
                    continue
                
                current_price = price.iloc[i]
                current_volume = df['volume'].iloc[i]
                
                # Calculate dynamic threshold
                dynamic_threshold = self._calculate_dynamic_threshold(
                    df, i, lookback, breakout_threshold
                )
                
                # Check for resistance breakout (strongest levels first)
                breakout_detected = False
                for resistance in resistance_levels:
                    if current_price > resistance['price'] * (1 + dynamic_threshold):
                        # Volume confirmation with strength weighting
                        volume_req = volume_ma.iloc[i] * volume_threshold * (1 + resistance['strength'] / 10)
                        if current_volume > volume_req:
                            events.append(df.index[i])
                            breakout_detected = True
                            break  # Only one event per bar
                
                if not breakout_detected:
                    # Check for support breakout (strongest levels first)
                    for support in support_levels:
                        if current_price < support['price'] * (1 - dynamic_threshold):
                            # Volume confirmation with strength weighting
                            volume_req = volume_ma.iloc[i] * volume_threshold * (1 + support['strength'] / 10)
                            if current_volume > volume_req:
                                events.append(df.index[i])
                                break  # Only one event per bar
            
            event_index = pd.DatetimeIndex(events)
            
            # Post-processing to avoid clustered events
            if len(event_index) > 1:
                event_index = self._post_process_events(event_index, pd.Timedelta(hours=2))
            
            # Log statistics
            n_resistance = len([r for r in resistance_levels if r['strength'] >= min_strength_score])
            n_support = len([s for s in support_levels if s['strength'] >= min_strength_score])
            logger.debug(f"SupportResistanceBreakEvents generated {len(event_index)} events "
                        f"(lookback={lookback}, min_touches={min_touches}, "
                        f"resistance_levels={n_resistance}, support_levels={n_support})")
            return event_index
            
        except Exception as e:
            logger.error(f"SupportResistanceBreakEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'breakout_threshold' in adjusted:
            adjusted['breakout_threshold'] *= factor
        if 'min_strength_score' in adjusted:
            adjusted['min_strength_score'] *= factor
        # Relax volume requirement
        if 'volume_threshold' in adjusted:
            adjusted['volume_threshold'] *= factor
        if 'lookback' in adjusted:
            # Shorter lookback = find local S/R more easily?
            adjusted['lookback'] = max(10, int(adjusted['lookback'] * factor))
        return adjusted

class OrderBlockEvents(BaseEventGenerator):
    """Generate events based on Order Block identification (Smart Money Concept).
    
    Order Blocks are the last up/down candle before a strong move,
    indicating institutional order placement zones.
    """
    
    def generate(self, df: pd.DataFrame, lookback: int = 20, min_move_pct: float = 0.5, 
                 volume_threshold: float = 2.0) -> pd.DatetimeIndex:
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"OrderBlockEvents requires OHLCV data. Missing: {[c for c in required_cols if c not in df.columns]}")
                return pd.DatetimeIndex([])
            
            if len(df) < lookback * 2:
                logger.warning(f"Insufficient data for Order Block analysis: need {lookback * 2}, got {len(df)}")
                return pd.DatetimeIndex([])
            
            events = []
            volume_ma = df['volume'].rolling(lookback).mean()
            
            # Scan for order blocks
            for i in range(lookback, len(df) - lookback):
                current_candle = df.iloc[i]
                
                # Check for bullish order block (last down candle before up move)
                if (current_candle['close'] < current_candle['open'] and  # Red candle
                    current_candle['volume'] > volume_ma.iloc[i] * volume_threshold):  # High volume
                    
                    # Check if strong up move follows
                    future_move = True
                    for j in range(i + 1, min(i + lookback, len(df))):
                        move_pct = (df.iloc[j]['close'] - current_candle['close']) / current_candle['close']
                        if move_pct > min_move_pct / 100:  # Convert percentage to decimal
                            future_move = True
                            break
                        elif move_pct < -min_move_pct / 100:  # Move against
                            future_move = False
                            break
                    
                    if future_move:
                        events.append(df.index[i])
                
                # Check for bearish order block (last up candle before down move)
                elif (current_candle['close'] > current_candle['open'] and  # Green candle
                      current_candle['volume'] > volume_ma.iloc[i] * volume_threshold):  # High volume
                    
                    # Check if strong down move follows
                    future_move = True
                    for j in range(i + 1, min(i + lookback, len(df))):
                        move_pct = (df.iloc[j]['close'] - current_candle['close']) / current_candle['close']
                        if move_pct < -min_move_pct / 100:  # Convert percentage to decimal
                            future_move = True
                            break
                        elif move_pct > min_move_pct / 100:  # Move against
                            future_move = False
                            break
                    
                    if future_move:
                        events.append(df.index[i])
            
            event_index = pd.DatetimeIndex(events)
            
            # Post-processing to avoid clustered events
            if len(event_index) > 1:
                event_index = self._post_process_events(event_index, pd.Timedelta(hours=6))
            
            logger.debug(f"OrderBlockEvents generated {len(event_index)} events (lookback={lookback}, min_move_pct={min_move_pct})")
            return event_index
            
        except Exception as e:
            logger.error(f"OrderBlockEvents generation failed: {e}")
            return pd.DatetimeIndex([])
    
    def _adjust_z_threshold(self, params: dict, factor: float) -> dict:
        adjusted = params.copy()
        if 'min_move_pct' in adjusted:
            adjusted['min_move_pct'] *= factor
        if 'volume_threshold' in adjusted:
             adjusted['volume_threshold'] *= factor
        if 'lookback' in adjusted:
             adjusted['lookback'] = max(10, int(adjusted['lookback'] * factor))
        return adjusted


# ==========================================
# 6. Final Diversity Filter
# ==========================================

def final_diversity_filter(
    geometries: List[OutputGeometry], 
    price: pd.Series,
    jaccard_threshold: float = 0.7,
    returns_threshold: float = 0.8
) -> List[OutputGeometry]:
    """
    Filter geometries to ensure diversity in both event timing AND returns patterns.
    
    Args:
        geometries: List of OutputGeometry objects (one per signal family)
        price: Price series for returns calculation
        jaccard_threshold: Maximum Jaccard similarity allowed (lower = more diverse)
        returns_threshold: Maximum returns correlation allowed (lower = more diverse)
    
    Returns:
        Filtered list of diverse geometries
    """
    if len(geometries) <= 1:
        return geometries
    
    logger.info(f"Applying final diversity filter to {len(geometries)} geometries...")
    
    # Sort by AUC score descending - keep highest scoring as anchor
    geometries.sort(key=lambda x: x.auc, reverse=True)
    
    # Build event timing indicators for Jaccard similarity
    event_indicators = {}
    for geo in geometries:
        indicator = build_indicator_matrix(geo.events, price.index, horizon=1).values.flatten().astype(bool)
        event_indicators[geo.name] = indicator
    
    # Calculate returns series for each geometry
    returns_series = {}
    for geo in geometries:
        if len(geo.events) > 0:
            # Calculate returns from event entry to horizon
            returns_list = []
            for event_time in geo.events:
                if event_time in price.index:
                    event_idx = price.index.get_loc(event_time)
                    horizon = min(120, len(price) - event_idx - 1)  # Use horizon from params or default
                    if horizon > 0:
                        start_price = price.iloc[event_idx]
                        end_price = price.iloc[min(event_idx + horizon, len(price) - 1)]
                        ret = (end_price - start_price) / start_price
                        returns_list.append(ret)
            
            if returns_list:
                returns_series[geo.name] = pd.Series(returns_list, index=geo.events[:len(returns_list)])
    
    # Diversity filtering
    selected = [geometries[0]]  # Always keep the highest scoring geometry
    rejected_log = []
    
    for candidate in geometries[1:]:
        is_diverse = True
        rejection_reasons = []
        
        for selected_geo in selected:
            # 1. Check event timing diversity (Jaccard)
            if candidate.name in event_indicators and selected_geo.name in event_indicators:
                candidate_indicator = event_indicators[candidate.name]
                selected_indicator = event_indicators[selected_geo.name]
                
                # Jaccard similarity = intersection / union
                intersection = np.logical_and(candidate_indicator, selected_indicator).sum()
                union = np.logical_or(candidate_indicator, selected_indicator).sum()
                jaccard_sim = intersection / union if union > 0 else 0
                
                if jaccard_sim > jaccard_threshold:
                    is_diverse = False
                    rejection_reasons.append(f"Jaccard similarity {jaccard_sim:.3f} > {jaccard_threshold} with {selected_geo.name}")
            
            # 2. Check returns pattern diversity
            if (candidate.name in returns_series and 
                selected_geo.name in returns_series and
                len(returns_series[candidate.name]) > 10 and 
                len(returns_series[selected_geo.name]) > 10):
                
                # Align returns on overlapping events
                candidate_returns = returns_series[candidate.name]
                selected_returns = returns_series[selected_geo.name]
                
                # Find overlapping events
                common_events = candidate_returns.index.intersection(selected_returns.index)
                if len(common_events) > 5:  # Need sufficient overlap
                    candidate_common = candidate_returns.loc[common_events]
                    selected_common = selected_returns.loc[common_events]
                    
                    # Calculate correlation
                    correlation = abs(np.corrcoef(candidate_common.values, selected_common.values)[0, 1])
                    
                    if not np.isnan(correlation) and correlation > returns_threshold:
                        is_diverse = False
                        rejection_reasons.append(f"Returns correlation {correlation:.3f} > {returns_threshold} with {selected_geo.name}")
        
        if is_diverse:
            selected.append(candidate)
            logger.info(f"✅ Selected {candidate.name} (AUC={candidate.auc:.3f})")
        else:
            rejected_log.append({
                'candidate': candidate.name,
                'auc': candidate.auc,
                'reasons': rejection_reasons
            })
            logger.info(f"❌ Rejected {candidate.name} (AUC={candidate.auc:.3f}): {'; '.join(rejection_reasons)}")
    
    logger.info(f"Final diversity filter: {len(selected)}/{len(geometries)} geometries retained")
    
    # Log rejection summary
    if rejected_log:
        logger.info("Rejection summary:")
        for log in rejected_log:
            logger.info(f"  - {log['candidate']}: {log['reasons'][0] if log['reasons'] else 'Unknown'}")
    
    return selected


# ==========================================
# 7. Enhanced Parameter Grids
# ==========================================

def get_enhanced_parameter_grids() -> Dict[str, Dict]:
    """
    Define enhanced parameter grids for each signal family including:
    - TP/SL ratios (expanded from fixed grid)
    - Horizons (multiple timeframes)
    - Lookback variations
    - MFE/MAE optimization parameters
    """
    
    # Enhanced TPSL grid (more granular)
    tpsl_grid = [
        # Conservative (high win rate)
        {'id': '1.5:1', 'pt': 1.5, 'sl': 1.0},
        {'id': '2:1', 'pt': 2.0, 'sl': 1.0},
        
        # Balanced
        {'id': '2.5:1', 'pt': 2.5, 'sl': 1.0},
        {'id': '3:1', 'pt': 3.0, 'sl': 1.0},
        {'id': '3.5:1', 'pt': 3.5, 'sl': 1.0},
        
        # Aggressive (high reward)
        {'id': '4:1', 'pt': 4.0, 'sl': 1.0},
    ]
    
    # Horizon options (Restricted to 12 and 48 per instruction)
    horizon_options = [12, 48]
    
    # Family-specific parameter grids (restricted to 12, 24, 48)
    family_grids = {
        'PRICE_CUSUM': {
            'base_params': [(0.5, 20), (0.75, 40)],  # multiplier, vol_window
        },
        'VOL_CUSUM': {
            'base_params': [(4.0, 100), (3.0, 50)],  # h, vol_span
        },
        'LIQ_CUSUM': {
            'base_params': [(4.0, 100), (3.0, 50)],  # h, vol_span
        },
        'VOL_PARTICIPATION': {
            'base_params': [(4.0, 100), (3.0, 50)],  # h, span
        },
    }
    
    return {
        'tpsl_grid': tpsl_grid,
        'horizon_options': horizon_options,
        'family_grids': family_grids,
    }

# ==========================================
# 8. Main Pipeline
# ==========================================

def _safe_to_markdown(df: pd.DataFrame) -> str:
    """Fallback for to_markdown() if tabulate is missing."""
    try:
        return df.to_markdown()
    except Exception:
        cols = df.columns
        res = [" | " + " | ".join(map(str, cols)) + " | "]
        res.append(" | " + " | ".join(["---"] * len(cols)) + " | ")
        for _, row in df.iterrows():
            formatted_row = [f"{x:.4f}" if isinstance(x, (float, np.float64, np.float32)) else str(x) for x in row]
            res.append(" | " + " | ".join(formatted_row) + " | ")
        return "\n".join(res)

def orthogonal_label_generation(
    data: Union[pd.Series, pd.DataFrame],
    volume: Optional[pd.Series] = None,
    df_full: Optional[pd.DataFrame] = None,
    target_signals_per_day: float = 7.5,
    use_adaptive_thresholds: bool = True
) -> List[OutputGeometry]:
    """
    Enhanced Execution Pipeline for Orthogonal Label Generation.
    Implements: Generate -> Score -> Top 50% -> Probe -> Final Diversity Filter.
    
    Replaced with 4 Orthogonal CUSUM Clocks.
    """
    tprint_info(f"--- Starting Advanced Geometry Generation (Target: {target_signals_per_day} signals/day) ---")

    # 0. Data Standardization
    if isinstance(data, pd.DataFrame):
        price = data['close']
        if volume is None and 'volume' in data.columns:
            volume = data['volume']
        if df_full is None:
            df_full = data
    else:
        price = data

    if volume is None and df_full is not None and 'volume' not in df_full.columns:
        volume = df_full['volume']
    
    # 1. Generate Probe Features
    X_probe = generate_probe_features(price, volume)
    
    # Use df_full if provided, else construct min necessary
    if df_full is None:
        df_full = pd.DataFrame({'close': price})
        if volume is not None:
            df_full['volume'] = volume
    elif 'volume' not in df_full.columns and volume is not None:
        df_full['volume'] = volume

    # 2. Get Enhanced Parameter Grids
    param_grids = get_enhanced_parameter_grids()
    
    # 3. Build Enhanced Candidate Configurations
    generator_configs = []
    
    # Replaced signal families with the 4 Orthogonal CUSUMs
    base_generators = [
        ('PRICE_CUSUM', AdaptiveSymmetricCUSUMEvents()),
        ('VOL_CUSUM', VolatilityCusumEvents()),
        ('LIQ_CUSUM', LiquidityCusumEvents()),
        ('VOL_PARTICIPATION', VolumeCusumEvents())
    ]
    
    # Build enhanced parameter combinations
    for fam, gen in base_generators:
        family_config = param_grids['family_grids'].get(fam, {})
        base_params = family_config.get('base_params', [])
        
        # Add base parameters only (no variations)
        for params in base_params:
            generator_configs.append((fam, gen, params))
    
    tprint_info(f"Generated {len(generator_configs)} enhanced candidate configurations")
    
    # 3. Process Candidates (Generate & Gate)
    candidates = []
    outcomes_log = []

    for fam, gen, params in generator_configs:

            try:
                # Use standard generation (Adaptive logic skipped for now for these new generators to match snippet)
                # But kept logic structure if needed.

                # Extended list of classes requiring DataFrame
                df_required = DF_REQUIRED_CLASSES + ('VolatilityCusumEvents', 'LiquidityCusumEvents', 'VolumeCusumEvents')

                if gen.__class__.__name__ in df_required:
                    events = gen.generate(df_full, *params)
                else:
                    events = gen.generate(price, *params)
            except Exception as e:
                tprint_warning(f"Generator {fam} failed: {e}")
                continue

            if len(events) < 5: 
                tprint_warning(f"Skipping {fam}: Too few events ({len(events)})")
                continue
            
            # Log signal rate for monitoring
            duration_days = (events[-1] - events[0]).days if len(events) > 1 else 1
            signals_per_day = len(events) / max(1, duration_days)
            tprint_info(f"DEBUG: {fam} generated {len(events)} events")

            # Create parameter dict for logging
            param_dict = {'params': params}

            # Iterate Enhanced Grids
            tpsl_grid = param_grids['tpsl_grid']
            horizon_options = param_grids['horizon_options']
            risk_budget_options = [0.4, 0.7, 1.0]  # 0=no drawdown before TP, 1=very close to SL on average
            
            for grid_item in tpsl_grid:
                pt = grid_item['pt']
                sl = grid_item['sl']
                
                for horizon in horizon_options:
                    for risk_budget in risk_budget_options:
                        high = df_full.get('high')
                        low = df_full.get('low')

                        if fam == 'PRICE_CUSUM':
                            # Classic Triple Barrier
                            labels, weights, returns, mfe, mae, vol = compute_dominance_labels(
                                price, events, df_full['volatility_1d'],
                                risk_budget=risk_budget, pt_mult=pt, sl_mult=sl, horizon=horizon,
                                high=high, low=low
                            )
                        elif fam == 'LIQ_CUSUM':
                            # Liquidity Stress -> Path Degradation
                            # d_sigma map from pt: pt=[1.5, 4.0] -> d=[0.75, 2.0]
                            d_sigma = pt * 0.5
                            labels, weights, returns, mfe, mae, vol = compute_path_degradation_labels(
                                df_full, events, horizon=horizon, d_sigma=d_sigma
                            )
                        else:
                            # Volatility & Volume -> Volatility Expansion
                            # k_factor map from pt: pt=[1.5, 4.0] -> k=[1.25, 1.5]
                            # User requested 1.2-1.5.
                            # Formula: 1.1 + pt * 0.1 -> 1.5=>1.25, 4.0=>1.5
                            k_factor = 1.1 + (pt * 0.1)
                            labels, weights, returns, mfe, mae, vol = compute_volatility_labels(
                                df_full, events, horizon=horizon, k=k_factor
                            )

                        if labels.empty:
                            continue

                        # Quality Checks - Only if labels exist
                        passed, metrics, status = check_label_quality(
                            events, labels, returns, df_full, X_probe, gen, param_dict
                        )

                        outcomes_log.append({
                            'family': fam,
                            'params': str(param_dict),
                            'pt_mult': pt,
                            'sl_mult': sl,
                            'horizon': horizon,
                            'risk_budget': risk_budget,
                            'status': status,
                            'n': metrics.get('n', 0),
                            'pos_rate': metrics.get('pos_rate', 0),
                            'min_p': metrics.get('min_p', 1.0),
                            'max_mi': metrics.get('max_mi', 0.0),
                            'signals_per_day': round(signals_per_day, 2),
                            'target_signals_per_day': target_signals_per_day,
                            'adaptive_used': False
                        })

                        if passed:
                            candidates.append({
                                'family': fam,
                                'events': events,
                                'labels': labels,
                                'weights': weights,
                                'returns': returns,
                                'mfe': mfe, 'mae': mae, 'vol': vol,
                                'params': {**param_dict, 'risk_budget': risk_budget, 'pt_mult': pt, 'sl_mult': sl, 'horizon': horizon},
                                'status': status
                            })

    # 4. Multi-Factor Scoring
    # 4. Multi-Factor Scoring
    scored_candidates = calculate_multifactor_score(candidates, X_probe)
    
    # Save outcomes log and Generate Gate Diagnostics Report
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "outcomes"
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Save CSV
        pd.DataFrame(outcomes_log).to_csv(f"{out_dir}/geometry_gates_{timestamp}.csv", index=False)
        tprint_info(f"Saved geometry gates log to {out_dir}/geometry_gates_{timestamp}.csv")
        
        # 2. Save Markdown Report
        diag_df = pd.DataFrame(outcomes_log)
        if not diag_df.empty:
            summary_lines = ["# Layer 2 Geometry Gate Diagnostics\n\n"]
            summary_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Family Stats
            summary_lines.append("## Family Pass Rates\n")
            family_stats = diag_df.groupby('family').agg({
                'status': lambda x: (x == 'PASS').mean()
            }).sort_values('status', ascending=False)
            summary_lines.append(_safe_to_markdown(family_stats) + "\n\n")
            
            # Failure Reasons
            summary_lines.append("## Top Failure Reasons\n")
            fail_df = diag_df[diag_df['status'] != 'PASS']
            if not fail_df.empty:
                fail_stats = fail_df['status'].value_counts().to_frame()
                summary_lines.append(_safe_to_markdown(fail_stats) + "\n\n")
            
            # Detail by Family
            summary_lines.append("## Failure Detail by Family\n")
            for fam in diag_df['family'].unique():
                fam_df = diag_df[diag_df['family'] == fam]
                top_fail = fam_df['status'].value_counts().index[0]
                summary_lines.append(f"- **{fam}**: Top status: {top_fail} (Pass rate: {(fam_df['status']=='PASS').mean():.1%})\n")
            
            diag_path = f"{out_dir}/layer2_gate_diagnostics_{timestamp}.md"
            with open(diag_path, 'w') as f:
                f.writelines(summary_lines)
            tprint_success(f"💾 Gate diagnostics report saved to {diag_path}")
            
    except Exception as e:
        logger.error(f"Failed to generate gate diagnostics: {e}")
    
    if not scored_candidates:
        tprint_warning("No candidates passed gates.")
        return []

    # Rank -> Top 50% -> Cluster (5) -> Top 1 -> Probe

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Keep Top 50%
    n_keep = max(1, len(scored_candidates) // 2)
    top_candidates = scored_candidates[:n_keep]
    logger.info(f"Top 50% selection: Kept {len(top_candidates)} from {len(scored_candidates)} candidates.")

    # 5. Run LGBM Probe on Top Candidates
    tprint_info(f"🚀 Running LGBM Probe on {len(top_candidates)} top candidates...")
    probe_geoms = []
    for i, cand in enumerate(top_candidates):
        tprint_info(f"🎯 Probing candidate {i+1}/{len(top_candidates)}: {cand['family']}_{cand['params']}")
        X = X_probe.loc[cand['labels'].index]
        metrics = run_lgbm_probe(X, cand['labels'], cand['weights'], cand['returns'])
        cand['metrics_probe'] = metrics

        # Create OutputGeometry
        indicator = build_indicator_matrix(cand['events'], price.index, horizon=120)
        purity = average_uniqueness(indicator)

        geo = OutputGeometry(
            name=f"{cand['family']}_{cand['params']}",
            family=cand['family'],
            events=cand['events'],
            labels=cand['labels'],
            weights=cand['weights'],
            purity=purity,
            auc=metrics.get('lift', 0.0),  # Store lift as primary metric
            params=cand['params'],
            metrics={**cand.get('metrics_raw', {}), **cand['metrics_probe']}
        )
        probe_geoms.append(geo)

    # 6. Apply Final Diversity Filter
    tprint_info(f"🌐 Applying Final Diversity Filter to {len(probe_geoms)} geometries...")
    final_geoms = final_diversity_filter(probe_geoms, price, 
                                       jaccard_threshold=0.7, 
                                       returns_threshold=0.8)

    tprint_info(f"🎉 Pipeline Complete: {len(final_geoms)} final geometries selected")
    return final_geoms


# ==========================================
# 7. Class Aliases for Backward Compatibility
# ==========================================
# Full implementations restored from commit 66a8da258

def generate_dual_cusum_signals(
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    k: float = 0.12,
    alpha: float = 1.0,
    beta: float = 1.0,
    er_min: float = 0.2,
    window_vol: int = 20,
    window_er: int = 10,
    Q: float = 1e-5,
    R: float = 0.01
) -> pd.DataFrame:
    """
    Generate dual CUSUM signals for trend-following and mean-reversion using Kalman filter.
    Restored from commit 66a8da258.
    """
    # 1. Compute log returns
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # 2. Apply 1D Kalman filter
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=float(log_ret.iloc[0]) if len(log_ret) > 0 else 0.0)
    log_ret_smooth_raw, _ = kf.filter_series(log_ret)

    if not isinstance(log_ret_smooth_raw, pd.Series):
        log_ret_smooth_series = pd.Series(log_ret_smooth_raw, index=close.index).fillna(0.0)
    else:
        log_ret_smooth_series = log_ret_smooth_raw.fillna(0.0)

    # 3. Rolling volatility & ER
    sigma = log_ret_smooth_series.rolling(window_vol, min_periods=1).std()
    change = log_ret_smooth_series.rolling(window_er).sum().abs()
    volatility = log_ret_smooth_series.abs().rolling(window_er, min_periods=1).sum()
    ER = (change / (volatility + 1e-12)).fillna(0.0)

    # 4. Liquidity & Thresholds
    liquidity_mod = pd.Series(1.0, index=close.index)
    if volume is not None:
        vol_ma = volume.rolling(window_vol, min_periods=1).mean()
        rel_volume = volume / (vol_ma + 1e-9)
        liquidity_mod = 1.0 + beta * (1.0 - rel_volume)
        liquidity_mod = liquidity_mod.clip(0.5, 2.0)

    regime_mod = 1.0 + alpha * (1.0 - ER)
    h_t = (k * sigma * regime_mod * liquidity_mod).fillna(0.0)

    # 5. Residuals for Reversal Logic
    expected_return = log_ret_smooth_series.rolling(window_vol, min_periods=1).mean()
    residual_ret = (log_ret_smooth_series - expected_return).fillna(0.0)

    # 6. CUSUM Loop
    n = len(close)
    r_arr = log_ret_smooth_series.to_numpy()
    res_arr = residual_ret.to_numpy()
    h_arr = h_t.to_numpy()
    er_arr = ER.to_numpy()

    trend_signal = np.zeros(n)
    reversal_signal = np.zeros(n)
    S_trend_pos, S_trend_neg = 0.0, 0.0
    S_rev_pos, S_rev_neg = 0.0, 0.0

    for t in range(n):
        if er_arr[t] < er_min:
            S_trend_pos, S_trend_neg = 0.0, 0.0
            S_rev_pos, S_rev_neg = 0.0, 0.0
            continue

        cur_h = h_arr[t]
        if np.isnan(cur_h) or cur_h <= 0:
            cur_h = 1e-4

        # Trend CUSUM
        S_trend_pos = max(0.0, S_trend_pos + r_arr[t])
        S_trend_neg = min(0.0, S_trend_neg + r_arr[t])

        if S_trend_pos > cur_h:
            trend_signal[t] = 1
            S_trend_pos = 0.0
        elif S_trend_neg < -cur_h:
            trend_signal[t] = -1
            S_trend_neg = 0.0

        # Reversal CUSUM
        S_rev_pos = max(0.0, S_rev_pos + res_arr[t])
        S_rev_neg = min(0.0, S_rev_neg + res_arr[t])

        if S_rev_pos > cur_h:
            reversal_signal[t] = 1
            S_rev_pos = 0.0
        elif S_rev_neg < -cur_h:
            reversal_signal[t] = -1
            S_rev_neg = 0.0

    return pd.DataFrame({
        'trend_signal': trend_signal,
        'reversal_signal': reversal_signal,
        'h_t': h_t,
        'er': ER
    }, index=close.index)


class AdaptiveSymmetricCUSUMEvents(BaseEventGenerator):
    """Symmetric CUSUM with Dynamic Thresholds based on Volatility."""
    def generate(self, price: pd.Series, multiplier: float = 0.5, vol_window: int = 20) -> pd.DatetimeIndex:
        t_events = []
        s_pos, s_neg = 0, 0
        diff = np.log(price).diff()
        vol = diff.rolling(vol_window).std()
        diff_val, vol_val, idx = diff.values, vol.values, price.index

        start_idx = vol_window
        if start_idx < len(vol_val) and np.isnan(vol_val[start_idx]):
            valid_indices = np.where(~np.isnan(vol_val))[0]
            start_idx = valid_indices[0] if len(valid_indices) > 0 else len(price)

        for i in range(start_idx, len(price)):
            h = vol_val[i] * multiplier
            if np.isnan(h) or h == 0:
                continue
            r = diff_val[i]
            if np.isnan(r):
                continue
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)
            if s_pos > h:
                s_neg = s_pos = 0
                t_events.append(idx[i])
            elif s_neg < -h:
                s_neg = s_pos = 0
                t_events.append(idx[i])
        return pd.DatetimeIndex(t_events)


class VolatilityCusumEvents(BaseEventGenerator):
    """
    VOLATILITY CUSUM — Risk Regime Change (Direction-Free).
    """
    def generate(self, df: pd.DataFrame, h: float = 4.0, vol_span: int = 100) -> pd.DatetimeIndex:
        # Handle input type (Series vs DF)
        if isinstance(df, pd.Series):
             # Try to reconstruct basic df if possible, but VolCUSUM needs close
             close = df
        else:
             close = df['close']

        # Logic from snippet
        # df["ret"] = np.log(df["close"]).diff()
        # df["vol"] = ewma_volatility(df["ret"], span=vol_span)
        # df["log_vol_change"] = np.log(df["vol"]).diff()

        ret = np.log(close).diff()
        vol = ewma_volatility(ret, span=vol_span)
        log_vol_change = np.log(vol).diff()

        s_pos, s_neg = 0.0, 0.0
        events = []

        # Iteration
        # Vectorization is hard for CUSUM, loop is fine
        vals = log_vol_change.values
        idx = close.index

        for t in range(1, len(vals)):
            x = vals[t]
            if np.isnan(x): continue

            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)

            if s_pos > h or abs(s_neg) > h:
                events.append(idx[t])
                s_pos, s_neg = 0.0, 0.0

        return pd.DatetimeIndex(events)

class LiquidityCusumEvents(BaseEventGenerator):
    """
    LIQUIDITY CUSUM — Market Stress / Liquidity Withdrawal.
    """
    def generate(self, df: pd.DataFrame, h: float = 4.0, vol_span: int = 100) -> pd.DatetimeIndex:
        if isinstance(df, pd.Series):
             # Needs High/Low
             logger.warning("LiquidityCusumEvents requires DataFrame with High/Low")
             return pd.DatetimeIndex([])

        ret = np.log(df["close"]).diff()
        vol = ewma_volatility(ret, span=vol_span)

        # Proxy for liquidity stress
        true_range = df["high"] - df["low"]
        # liq_stress = np.log(true_range / vol)
        # Handle potential zeros/nans
        liq_stress = np.log(true_range / (vol + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0)

        s_pos, s_neg = 0.0, 0.0
        events = []

        vals = liq_stress.values
        idx = df.index

        for t in range(1, len(vals)):
            x = vals[t]
            if np.isnan(x): continue

            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)

            if s_pos > h or abs(s_neg) > h:
                events.append(idx[t])
                s_pos, s_neg = 0.0, 0.0

        return pd.DatetimeIndex(events)

class VolumeCusumEvents(BaseEventGenerator):
    """
    VOLUME CUSUM — Participation Shock.
    """
    def generate(self, df: pd.DataFrame, h: float = 4.0, span: int = 100) -> pd.DatetimeIndex:
        if isinstance(df, pd.Series):
            # Assumes series is volume? Or Price?
            # User snippet uses 'volume' column
            if df.name and 'volume' in df.name.lower():
                volume = df
            else:
                logger.warning("VolumeCusumEvents requires volume data")
                return pd.DatetimeIndex([])
        else:
            if 'volume' not in df.columns:
                 logger.warning("VolumeCusumEvents requires volume data")
                 return pd.DatetimeIndex([])
            volume = df['volume']

        vol_avg = volume.ewm(span=span, adjust=False).mean()
        # vol_surprise = np.log(volume / vol_avg)
        vol_surprise = np.log(volume / (vol_avg + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0)

        s_pos, s_neg = 0.0, 0.0
        events = []

        vals = vol_surprise.values
        idx = volume.index

        for t in range(1, len(vals)):
            x = vals[t]
            if np.isnan(x): continue

            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)

            if s_pos > h or abs(s_neg) > h:
                events.append(idx[t])
                s_pos, s_neg = 0.0, 0.0

        return pd.DatetimeIndex(events)


class SymmetricCusumEvents(BaseEventGenerator):
    """Simple symmetric CUSUM with fixed threshold."""
    def generate(self, price: pd.Series, h: float = 0.01) -> pd.DatetimeIndex:
        t_events = []
        s_pos, s_neg = 0, 0
        diff = np.log(price).diff().dropna()
        for i in diff.index:
            r = diff.loc[i]
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)
            if s_pos > h:
                s_neg = s_pos = 0
                t_events.append(i)
            elif s_neg < -h:
                s_neg = s_pos = 0
                t_events.append(i)
        return pd.DatetimeIndex(t_events)


class ImprovedCUSUMEvents(BaseEventGenerator):
    """CUSUM with differentiated trend vs reversal weights."""
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        k = params.get('k', 0.12)
        vol_window = params.get('vol_window', 20)
        er_window = params.get('er_window', 10)
        er_min = params.get('er_min', 0.2)
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        w_trend = params.get('w_trend', 1.0)
        w_reversal = params.get('w_reversal', 1.0)

        close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        volume = df.get('volume') or df.get('Volume')

        try:
            dual_signals = generate_dual_cusum_signals(
                close=close, volume=volume, k=k, alpha=alpha, beta=beta,
                er_min=er_min, window_vol=vol_window, window_er=er_window
            )
            composite = w_trend * dual_signals['trend_signal'] + w_reversal * dual_signals['reversal_signal']
            return composite.index[composite != 0]
        except Exception as e:
            logger.warning(f"ImprovedCUSUM failed: {e}")
            return pd.DatetimeIndex([])


class VolatilityShockEvents(BaseEventGenerator):
    """Detects volatility shocks."""
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        vol_mean = vol.expanding(min_periods=lookback).mean()
        vol_std = vol.expanding(min_periods=lookback).std()
        zscore = (vol - vol_mean) / (vol_std + 1e-6)
        return price.index[zscore > z]


class TrendInitiationEvents(BaseEventGenerator):
    """Detects trend initiations via MA crossover."""
    def generate(self, price: pd.Series, short: int = 20, long: int = 100) -> pd.DatetimeIndex:
        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()
        cross = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
        return price.index[cross]


class MeanReversionExtremeEvents(BaseEventGenerator):
    """Detects mean reversion extremes via z-score."""
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        mean = price.rolling(lookback).mean()
        std = price.rolling(lookback).std()
        zscore = (price - mean) / (std + 1e-6)
        return price.index[np.abs(zscore) > z]


class LiquidityShockEvents(BaseEventGenerator):
    """Detects liquidity shocks via volume z-score."""
    def generate(self, volume: pd.Series, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        vol_mean = volume.expanding(min_periods=lookback).mean()
        vol_std = volume.expanding(min_periods=lookback).std()
        zscore = (volume - vol_mean) / (vol_std + 1e-6)
        return volume.index[zscore > z]


class TimeEvents(BaseEventGenerator):
    """Time-based periodic events."""
    def generate(self, price: pd.Series, frequency: int = 24) -> pd.DatetimeIndex:
        return price.index[::frequency]


# Aliases
CusumEvents = AdaptiveSymmetricCUSUMEvents
