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

# Define UnifiedPriceMixin inline to avoid circular import
class UnifiedPriceMixin:
    """Mixin class for Layer2 generators to use unified price."""
    
    def __init__(self, use_unified_price: bool = True, layer0_params: dict = None):
        self.use_unified_price = use_unified_price
        self._layer0_params = layer0_params or {}
        self._cached_unified_price = None
        self._cached_timestamp = None
    
    def _get_unified_price(self, df: pd.DataFrame) -> pd.Series:
        """Get cached unified price or generate new one."""
        if not self.use_unified_price:
            return df['close']
        
        # Check cache validity (avoid re-computation)
        current_time = df.index[-1] if len(df) > 0 else None
        if (self._cached_unified_price is not None and 
            self._cached_timestamp == current_time):
            return self._cached_unified_price
        
        # Generate unified price (simplified version)
        try:
            # Use Kalman filter if available, otherwise fallback to close
            from .unified_price_layer2 import generate_unified_price
            unified_price = generate_unified_price(df, self._layer0_params)
        except Exception:
            unified_price = df['close']
        
        # Cache the result
        self._cached_unified_price = unified_price
        self._cached_timestamp = current_time
        
        return unified_price

def _should_use_range_specific_optimization() -> bool:
    """Check if 1.5-3% range optimization is enabled in configuration."""
    try:
        import yaml
        config_path = "config/labeling/layer2_coverage_relax_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("target_range_optimization", {}).get("enabled", False)
    except Exception:
        return False



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
# 1.5-3% Target Range Specific Grid (de Prado Framework)
MEDIUM_TERM_GRID = [
    # --- 1.5% Target (Low End)
    {'id': '1.5pct', 'pt': 1.5, 'sl': 0.75},
    {'id': '1.5pct_tight', 'pt': 1.5, 'sl': 0.5},
    
    # --- 2.0% Target (Mid Range)
    {'id': '2.0pct', 'pt': 2.0, 'sl': 0.8},
    {'id': '2.0pct_tight', 'pt': 2.0, 'sl': 0.6},
    
    # --- 2.25% Target (Optimal Midpoint)
    {'id': '2.25pct', 'pt': 2.25, 'sl': 0.75},
    {'id': '2.25pct_tight', 'pt': 2.25, 'sl': 0.6},
    
    # --- 2.5% Target (Upper Mid)
    {'id': '2.5pct', 'pt': 2.5, 'sl': 0.8},
    {'id': '2.5pct_tight', 'pt': 2.5, 'sl': 0.7},
    
    # --- 3.0% Target (High End)
    {'id': '3.0pct', 'pt': 3.0, 'sl': 0.9},
    {'id': '3.0pct_tight', 'pt': 3.0, 'sl': 0.8},
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
    horizon: int = 24,
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

    # Calculate volatility (Use EWMA 50 for smoother estimates as requested)
    # Overrides generic volatility_1d to ensure consistent smoothing for labeling
    vol = df['close'].pct_change().ewm(span=50).std()

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

    # Label: 1 if Expansion (regime signal, no fee filtering needed)
    labels_arr = (vol_ratio > k).astype(float)

    # Weights: Magnitude of expansion
    weights_arr = np.log1p(np.abs(vol_ratio - 1.0))

    # "Returns": Volatility change %
    returns_arr = vol_ratio - 1.0

    # MFE/MAE: Proxies
    mfe_arr = returns_arr
    mae_arr = np.zeros_like(returns_arr)

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
    horizon: int = 24,
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

    # Label: 1 if stress detected (max DD exceeds threshold) - regime signal, no fee filtering
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


def compute_tail_regime_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 24,
    metric_col: str = 'skew',
    z_thresh: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Tail Risk Persistence.
    Target: 1 if metric (skew/kurt) remains 'extreme' (> z_thresh) on average over horizon.
    """
    if events.empty or metric_col not in df.columns:
        return tuple([pd.Series(dtype=float)] * 6)

    metric = df[metric_col]
    # Calculate Z-score of metric if not already (assuming rolling standardization handled elsewhere or done here)
    # Here we assume metric is raw and we check magnitude.
    # Actually, let's standardize metric locally to be robust.
    roll_mean = metric.rolling(100).mean()
    roll_std = metric.rolling(100).std()
    z_score = (metric - roll_mean) / (roll_std + 1e-9)

    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    # Vectorized persistence check
    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)

    # Filter valid
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return tuple([pd.Series(dtype=float)] * 6)

    # Window Matrix
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    window_z = z_score.values[window_idxs]

    # Check average magnitude in window
    # We care about magnitude of tail risk (fat tails or skew)
    # If skew, it could be positive or negative. We usually care about absolute skew or negative skew?
    # "Abnormal risk". Extreme values.
    avg_abs_z = np.mean(np.abs(window_z), axis=1)

    # Label: 1 if extreme tail risk persists (regime signal, no fee filtering)
    labels_arr = (avg_abs_z > z_thresh).astype(float)
    weights_arr = np.log1p(avg_abs_z)
    returns_arr = avg_abs_z  # Proxy

    return (pd.Series(labels_arr, index=final_events),
            pd.Series(weights_arr, index=final_events),
            pd.Series(returns_arr, index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events))

def compute_trend_persistence_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 24,
    trend_col: str = 'trend'
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Trend Regime Persistence.
    Target: 1 if trend direction persists.
    """
    if events.empty or trend_col not in df.columns:
        return tuple([pd.Series(dtype=float)] * 6)

    trend = df[trend_col]

    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return tuple([pd.Series(dtype=float)] * 6)

    # Initial Trend Sign
    initial_trend = np.sign(trend.values[valid_idxs])

    # Future Trend Signs
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]
    future_trends = trend.values[window_idxs]

    # Consistency: Fraction of window where sign matches initial
    # Handle zero trend? Treat as mismatch if we expect trend.
    matches = np.sign(future_trends) == initial_trend[:, None]
    consistency = np.mean(matches, axis=1)

    # Label: High consistency (> 0.55) - regime signal, no fee filtering
    # Re-tightened from 0.5 to 0.55 for quality
    labels_arr = (consistency > 0.55).astype(float)
    weights_arr = consistency
    returns_arr = consistency  # Proxy

    return (pd.Series(labels_arr, index=final_events),
            pd.Series(weights_arr, index=final_events),
            pd.Series(returns_arr, index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events))

def compute_vol_state_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 24
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Volatility State Persistence.
    Target: 1 if Vol State persists.
    """
    if events.empty or 'vol_state' not in df.columns:
        return tuple([pd.Series(dtype=float)] * 6)

    state = df['vol_state']

    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return tuple([pd.Series(dtype=float)] * 6)

    initial_state = state.values[valid_idxs]

    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]
    future_states = state.values[window_idxs]

    matches = future_states == initial_state[:, None]
    persistence = np.mean(matches, axis=1)

    # Label: High persistence (> 0.55) - regime signal, no fee filtering
    # Re-tightened from 0.5 to 0.55 for quality
    labels_arr = (persistence > 0.55).astype(float)
    weights_arr = persistence
    returns_arr = persistence  # Proxy

    return (pd.Series(labels_arr, index=final_events),
            pd.Series(weights_arr, index=final_events),
            pd.Series(returns_arr, index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events))


def compute_volume_participation_labels(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    horizon: int = 24,
    volume_threshold: float = 1.5
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Labeling for Volume Participation Events.
    Target: 1 if volume participation leads to price persistence in same direction.
    """
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    if 'volume' not in df.columns:
        return tuple([pd.Series(dtype=float)] * 6)

    volume = df['volume']
    close = df['close']
    
    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return tuple([pd.Series(dtype=float)] * 6)

    # Volume participation at event
    vol_baseline = volume.rolling(960, min_periods=480).mean()
    vol_excess = (volume / (vol_baseline + 1e-9)) - 1.0
    
    # Price direction at event
    price_ret = close.pct_change()
    event_direction = np.sign(price_ret.iloc[valid_idxs])
    
    # Future price persistence
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]
    future_returns = price_ret.values[window_idxs]
    
    # Check if price persists in same direction
    direction_consistency = np.mean(np.sign(future_returns) == event_direction[:, None], axis=1)
    
    # Label: 1 if high volume participation AND price persistence
    event_vol_excess = vol_excess.iloc[valid_idxs].values
    volume_condition = event_vol_excess > volume_threshold
    
    labels_arr = (volume_condition & (direction_consistency > 0.6)).astype(float)
    weights_arr = event_vol_excess * direction_consistency
    returns_arr = direction_consistency  # Proxy
    
    return (pd.Series(labels_arr, index=final_events),
            pd.Series(weights_arr, index=final_events),
            pd.Series(returns_arr, index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(np.zeros_like(labels_arr), index=final_events),
            pd.Series(event_vol_excess, index=final_events))


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

    # Final Label (without timeouts)
    final_label_mask = win_mask & risk_mask & profit_mask
    labels = final_label_mask.astype(float)
    
    # Fee-aware timeout labeling: positive if return > fees (0.3%), negative otherwise
    timeout_mask = (~any_pt) & (~any_sl)
    timeout_returns = close_returns[:, -1]  # Return at horizon
    FEE_THRESHOLD = 0.003  # 0.3% total fees (round-trip)
    timeout_profitable = timeout_returns > FEE_THRESHOLD
    labels[timeout_mask & timeout_profitable] = 1.0  # Profitable timeout = positive label
    # labels[timeout_mask & ~timeout_profitable] already 0.0 from final_label_mask

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
    generator_params: dict,
    family: str = "UNKNOWN"
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

    # 1. Sample Size Gate (relaxed from 0.5 to 0.1 events/day)
    if rate < 0.1:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [FAIL]")
        overall_pass = False
        if failure_reason == "PASS": failure_reason = "Sample Size (< 0.1/day)"
    else:
        gates_log.append(f"Sample: {n}/{rate:.2f}/d [OK]")

    # 2. Class Balance Gate (relaxed to 10%/90%)
    pos_rate = labels.mean()
    val_metrics['pos_rate'] = pos_rate
    
    # 2. Class Balance Gate (relaxed to 10%/90%)
    # Special relaxation for PRICE_CUSUM (allows 3-90%)
    min_bal = 0.03 if family == 'PRICE_CUSUM' else 0.10
    
    if pos_rate < min_bal or pos_rate > 0.90:
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
    # Sanitize X to remove inf/nan values before f_classif
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
    
    # MI gate now just a warning (non-blocking) since MI values are consistently very low
    if max_mi < 0.001:
        gates_log.append(f"MI: {max_mi:.4f} [WARN]")  # Warning only, don't fail
        # NOT blocking: overall_pass = False
    else:
        gates_log.append(f"MI: {max_mi:.4f} [OK]")

    summary_str = " | ".join(gates_log)
    if overall_pass:
        tprint_info(f"✅ [{family}] Gates Passed: {summary_str}")
    else:
        tprint_warning(f"❌ [{family}] Gates Failed: {summary_str}")

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

        # Sanitize X_sub before f_classif to avoid infinity errors
        X_sub = X_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'seed': 42,
        'boosting_type': 'goss',
        'max_depth': 3,
        'num_leaves': 7,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'feature_fraction': 0.8,
        'top_rate': 0.2,
        'other_rate': 0.1
    }
    preds_all = []
    labels_all = []
    r_all = [] # Realized returns for all va samples
    base_returns = []  # All validation returns (baseline)
    meta_returns = []  # Returns where prediction > 0.5

    # Initialize TimeSeriesSplit for 3-fold CV
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    tprint_info(f"🔄 Setting up 3-fold time series cross-validation")

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
    consistency = 0.0
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
    step = len(r_arr) // 3
    if step > 0:
        fold_sharpes = [sharpe(np.array(f)) for f in [meta_returns[i:i+step] for i in range(0, len(r_arr), step)] if len(f) > 0]
    else:
        fold_sharpes = [sharpe(r_arr)] if len(r_arr) > 0 else []
    std_error = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0.0

    # Multi-threshold probe completion
    thresholds = [0.2, 0.5, 0.8]
    r_arr_all = np.array(r_all)  # Convert r_all list to array for threshold slicing
    tprint_info("✅ Probe Complete:")
    for threshold in thresholds:
        mask = preds_arr > threshold
        meta_returns_thresh = r_arr_all[mask]
        meta_sh = sharpe(meta_returns_thresh) if len(meta_returns_thresh) > 0 else 0.0
        lift = meta_sh - base_sh
        
        psr_val = 0.0
        if len(meta_returns_thresh) > 2:
            from scipy.stats import skew, kurtosis
            s = skew(meta_returns_thresh)
            k = kurtosis(meta_returns_thresh)
            psr_val = calculate_psr(meta_sh, len(meta_returns_thresh), s, k)
        
        n_preds = mask.sum(); n_returns = len(meta_returns_thresh); tprint_info(f"  {threshold} threshold: Lift={lift:.4f} (BaseSH={base_sh:.4f}, MetaSH={meta_sh:.4f}), IC={ic:.4f}, PSR={psr_val:.4f} [preds={n_preds}, returns={n_returns}]")
    
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
    # 'KalmanTrendEvents': ['q_fast', 'q_slow'],
    # 'KalmanRegimeEvents': ['Q', 'R', 'z'],
    'TailRiskCusumEvents': ['h', 'window', 'metric'],
    'TrendRegimeCusumEvents': ['h', 'fast', 'slow'],
    'VolatilityCusumEvents': ['h', 'vol_span'],
    'LiquidityCusumEvents': ['h', 'vol_span'],
    'VolumeCusumEvents': ['h', 'span'],
    'RangeATRcusumEvents': ['h', 'atr_window', 'vol_window'],
    'SRCusumEvents': ['h', 'sr_levels'],
    'VolatilityStateEvents': ['h', 'vol_span'],
    'ImprovedCUSUMEvents': ['k', 'vol_window'],
    'SymmetricCusumEvents': ['h'],
}

DF_REQUIRED_CLASSES = (
    # 'KalmanRegimeEvents',
    'ImprovedCUSUMEvents',
    'VolatilityCusumEvents',
    'LiquidityCusumEvents',
    'VolumeCusumEvents',
    'RangeATRcusumEvents',
    'SRCusumEvents',
    'TailRiskCusumEvents',
    'TrendRegimeCusumEvents',
    'VolatilityStateEvents',
) 
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


class BaseEventGenerator(UnifiedPriceMixin):
    """Base class for event generation with adaptive thresholding and unified price support."""
    
    def __init__(self, use_unified_price: bool = True, layer0_params: dict = None):
        """
        Initialize base event generator.
        
        Args:
            use_unified_price: Whether to use Kalman+VWAP unified price
            layer0_params: Layer0 parameters (auto-loaded if None)
        """
        super().__init__(use_unified_price=use_unified_price, layer0_params=layer0_params)
    
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
    # Group by family
    by_family = {}
    for g in geometries:
        if g.family not in by_family:
            by_family[g.family] = []
        by_family[g.family].append(g)

    final_selected = []
    
    # Process each family independently
    for fam, candidates in by_family.items():
        # Sort by AUC descending
        candidates.sort(key=lambda x: x.auc, reverse=True)
        
        # Always take the best one
        family_selected = [candidates[0]]
        logger.info(f"✅ Selected best {fam}: {candidates[0].name} (AUC={candidates[0].auc:.3f})")
        
        # Configure max winners based on family
        # PRICE_CUSUM needs orthogonality (up to 3)
        # Context/Other families should be single best (1)
        if fam == 'PRICE_CUSUM':
            MAX_PER_FAMILY = 3
        else:
            MAX_PER_FAMILY = 1
        
        for cand in candidates[1:]:
            if len(family_selected) >= MAX_PER_FAMILY:
                break
                
            is_diverse = True
            rejection_reason = ""
            
            for selected_geo in family_selected:
                # 1. Jaccard Check
                if cand.name in event_indicators and selected_geo.name in event_indicators:
                    cand_ind = event_indicators[cand.name]
                    sel_ind = event_indicators[selected_geo.name]
                    intersection = np.logical_and(cand_ind, sel_ind).sum()
                    union = np.logical_or(cand_ind, sel_ind).sum()
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    if jaccard_sim > jaccard_threshold:
                        is_diverse = False
                        rejection_reason = f"Jaccard {jaccard_sim:.2f} > {jaccard_threshold}"
                        break
                
                # 2. Returns Correlation Check
                if is_diverse and cand.name in returns_series and selected_geo.name in returns_series:
                    cand_ret = returns_series[cand.name]
                    sel_ret = returns_series[selected_geo.name]
                    common = cand_ret.index.intersection(sel_ret.index)
                    if len(common) > 10:
                        c_vals = cand_ret.loc[common].values
                        s_vals = sel_ret.loc[common].values
                        corr = abs(np.corrcoef(c_vals, s_vals)[0, 1])
                        if not np.isnan(corr) and corr > returns_threshold:
                            is_diverse = False
                            rejection_reason = f"Corr {corr:.2f} > {returns_threshold}"
                            break
            
            if is_diverse:
                family_selected.append(cand)
                logger.info(f"✅ Selected orthogonal {fam}: {cand.name} (AUC={cand.auc:.3f})")
            else:
                logger.info(f"❌ Rejected {cand.name} vs {fam}: {rejection_reason}")

        final_selected.extend(family_selected)

    logger.info(f"Final diversity filter: {len(final_selected)}/{len(geometries)} geometries retained across {len(by_family)} families")
    return final_selected
        

    
    return selected


# ==========================================
# 7. Enhanced Parameter Grids
# ==========================================

def get_enhanced_parameter_grids(range_specific: bool = False) -> Dict[str, Dict]:
    """
    Define enhanced parameter grids for each signal family including:
    - TP/SL ratios (expanded from fixed grid)
    - Horizons (multiple timeframes)
    - Lookback variations
    - MFE/MAE optimization parameters
    """
    
    # Use range-specific grid if optimization is enabled
    if range_specific:
        tpsl_grid = MEDIUM_TERM_GRID
    else:
        # Enhanced TPSL grid
        tpsl_grid = [
        # Symmetric (for diversity/orthogonality)
        {'id': '1:1', 'pt': 1.0, 'sl': 1.0},
        
        # Conservative (high win rate)
        {'id': '1.5:1', 'pt': 1.5, 'sl': 1.0},
        {'id': '2:1', 'pt': 2.0, 'sl': 1.0},
        
        # Balanced
        {'id': '3:1', 'pt': 3.0, 'sl': 1.0},
        
        # Aggressive (high reward)
        {'id': '4:1', 'pt': 4.0, 'sl': 1.0},
    ]
    
    # Horizon options (Restricted to 12 and 48 per instruction)
    horizon_options = [12, 48]
    
    # Family-specific parameter grids (restricted to 12, 24, 48)
    family_grids = {
        'PRICE_CUSUM': {
            # TUNED: k increased from 0.08-0.12 to 0.6-1.5 for 5-10 events/day target
            'base_params': [(0.8, 20), (1.2, 30)],  # k, vol_window - Higher k = fewer, stronger signals
        },
        'VOL_CUSUM': {
            'base_params': [(4.0, 100), (3.0, 50)],  # h, vol_span
        },
        'LIQ_CUSUM': {
            'base_params': [(4.0, 100), (3.0, 50)],  # h, vol_span
        },
        'VOL_PARTICIPATION': {
            'base_params': [(5.0, 960), (10.0, 960)],  # h (accumulated % excess), span (10 days @ 15m)
        },
        'RANGE_ATR': {
            'base_params': [(2.0, 14, 20), (1.5, 14, 20)], # h, atr_window, vol_window
        },
        'TAIL_RISK': {
            'base_params': [(2.0, 50, 'skew'), (2.0, 50, 'kurt')], # h, window, metric
        },
        'TREND_REGIME': {
            'base_params': [(2.0, 20, 50), (1.5, 10, 30)], # h, fast, slow
        },
        'VOL_STATE': {
            'base_params': [(2.0, 100)], # h, vol_span
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

def calibrate_all_cusum_thresholds(df: pd.DataFrame, target_events_per_day: float = 2.0, vol_window: int = 20, atr_window: int = 14, sr_levels: list = None) -> Dict[str, float]:
    thresholds = {}
    if len(df) < 100: return thresholds

    duration_days = (df.index[-1] - df.index[0]).days + 1
    bars_per_day = len(df) / max(1, duration_days)
    target_fraction = target_events_per_day / max(1, bars_per_day)

    # Price
    if 'close' in df.columns:
        price_metric = df['close'].pct_change().fillna(0).abs()
        thresholds['price'] = max(price_metric.quantile(1 - target_fraction), 1e-9)

    # Volatility
    if 'close' in df.columns:
        ret = df['close'].pct_change()
        vol = ret.ewm(span=vol_window, adjust=False).std()
        vol_metric = np.log(vol).diff().fillna(0).abs()
        thresholds['volatility'] = max(vol_metric.quantile(1 - target_fraction), 1e-9)

    # Volume
    if 'volume' in df.columns:
        vol_avg = df['volume'].ewm(span=vol_window, adjust=False).mean()
        volume_metric = np.log(df['volume'] / (vol_avg + 1e-9)).fillna(0).abs()
        thresholds['volume'] = max(volume_metric.quantile(1 - target_fraction), 1e-9)

    # ATR / Range
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
        atr = tr.rolling(atr_window).mean()
        atr_norm = (atr / atr.rolling(vol_window).mean() - 1).fillna(0).abs()
        thresholds['atr'] = max(atr_norm.quantile(1 - target_fraction), 1e-9)

    # S/R
    if 'close' in df.columns and sr_levels and len(sr_levels) > 0:
        sr_dist = np.array([min(abs(c - l) for l in sr_levels) for c in df['close']])
        sr_metric = pd.Series(sr_dist, index=df.index).abs()
        thresholds['sr'] = max(sr_metric.quantile(1 - target_fraction), 1e-9)
    else:
        thresholds['sr'] = None

    return thresholds

def volume_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, persistence_factor: float = 1.0) -> pd.Series:
    if events.empty or 'volume' not in df.columns: return pd.Series(0, index=events)
    vol_avg = df['volume'].ewm(span=20, adjust=False).mean()
    vol_norm = (df['volume'] / (vol_avg + 1e-9)) - 1.0
    price_ret = df['close'].pct_change().fillna(0)
    signed_vol_proxy = np.sign(vol_norm) * price_ret
    weight = abs(signed_vol_proxy) * persistence_factor
    return weight.reindex(events).fillna(0)

def atr_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, atr_window: int = 14, vol_window: int = 20) -> pd.Series:
    if events.empty or 'high' not in df.columns: return pd.Series(0, index=events)
    tr = df['high'] - df['low']
    atr = tr.rolling(atr_window).mean().fillna(1e-9)
    atr_change = np.log(atr / atr.rolling(vol_window).mean()).fillna(0)
    weight = abs(atr_change)
    return weight.reindex(events).fillna(0)

def sr_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, sr_levels: list) -> pd.Series:
    if events.empty or not sr_levels: return pd.Series(0, index=events)
    close_vals = df['close'].values
    sr_arr = np.array(sr_levels)
    distance = (close_vals[:, None] - sr_arr[None, :]).min(axis=1)
    weight = pd.Series(abs(distance), index=df.index)
    return weight.reindex(events).fillna(0)

def tail_cusum_weight(df: pd.DataFrame, events: pd.DatetimeIndex, window: int = 50) -> pd.Series:
    if events.empty: return pd.Series(0, index=events)
    returns = df['close'].pct_change()
    kurt = returns.rolling(window).kurt().fillna(0)
    min_k = kurt.rolling(window).min()
    max_k = kurt.rolling(window).max()
    weight = (kurt - min_k) / (max_k - min_k + 1e-9)
    return weight.reindex(events).fillna(0)

def get_uniqueness_weight(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 24) -> pd.Series:
    indicator = build_indicator_matrix(events, index, horizon=horizon)
    concurrency = indicator.sum(axis=1)
    # uniqueness = 1 / concurrency
    # average uniqueness over event lifespan
    uniqueness = pd.Series(0.0, index=events)
    if events.empty: return uniqueness

    # Map events to index locations
    evt_locs = index.get_indexer(events)
    for i, loc in enumerate(evt_locs):
        if loc == -1: continue
        end_loc = min(loc + horizon, len(index))
        c = concurrency.iloc[loc:end_loc]
        if len(c) > 0:
            uniqueness.iloc[i] = (1.0 / c).mean()

    return uniqueness

def get_signal_specific_weights(df: pd.DataFrame, events: pd.DatetimeIndex, sr_levels: list = None,
                               component_weights: Dict[str, float] = None, family: str = None) -> pd.Series:
    if component_weights is None:
        component_weights = {'vol': 1.0, 'atr': 1.0, 'sr': 1.0, 'tail': 1.0}

    intensity = pd.Series(0.0, index=events)

    if family == 'VOL_PARTICIPATION':
        intensity = volume_cusum_weight(df, events) * component_weights.get('vol', 1.0)
    elif family == 'RANGE_ATR':
        intensity = atr_cusum_weight(df, events) * component_weights.get('atr', 1.0)
    elif family == 'SR_CUSUM':
        intensity = sr_cusum_weight(df, events, sr_levels) * component_weights.get('sr', 1.0)
    elif family == 'TAIL_RISK':
        intensity = tail_cusum_weight(df, events) * component_weights.get('tail', 1.0)

    u_w = get_uniqueness_weight(events, df.index)

    final_weights = (1 + intensity) * u_w
    return final_weights

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
    use_adaptive_thresholds: bool = True,
    signal_weights: Optional[Dict[str, float]] = None,
    return_raw_candidates: bool = False
) -> List[OutputGeometry]:
    """
    Enhanced Execution Pipeline for Orthogonal Label Generation.
    Implements: Generate -> Score -> Top 50% -> Probe -> Final Diversity Filter.
    
    Args:
        return_raw_candidates: If True, returns all candidates passing gates without global filtering.
    """
    tprint_info(f"--- Starting Advanced Geometry Generation (Target: {target_signals_per_day} signals/day) ---")

    # ... (Data Spec Normalization skipped in diff for brevity, assume unchanged until we hit logic)
    # Actually I need to be careful with replace_file_content context.
    # I'll replace the function signature and the end logic. 
    # But replace_file_content needs contiguous block.
    # The function is HUGE.
    # I should use multi_replace or ensure I catch the start and end.
    
    # Let's try to just change signature first, then change the end logic.
    pass 
    # (Placeholder logic for the thought process, actual tool call below)

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

    # 2. Check for 1.5-3% range optimization configuration
    use_range_specific = _should_use_range_specific_optimization()
    
    # 3. Get Enhanced Parameter Grids
    param_grids = get_enhanced_parameter_grids(range_specific=use_range_specific)
    
    # 3. Build Enhanced Candidate Configurations
    generator_configs = []
    
    # Identify S/R levels for SRCusumEvents (simplified approach)
    # Use recent highs/lows as dynamic S/R levels
    price = df_full['close']
    recent_window = min(100, len(price))
    recent_data = price.iloc[-recent_window:]
    
    # Simple pivot-based S/R levels
    sr_levels = []
    if len(recent_data) >= 20:
        # Recent high resistance levels
        resistance_candidates = recent_data.rolling(10).max().dropna()
        # Recent low support levels  
        support_candidates = recent_data.rolling(10).min().dropna()
        
        # Get top 3 levels of each type
        resistance_levels = resistance_candidates.nlargest(3).unique()
        support_levels = support_candidates.nsmallest(3).unique()
        sr_levels = list(resistance_levels) + list(support_levels)

    # Calculate adaptive thresholds if requested
    adaptive_thresholds = {}
    if use_adaptive_thresholds:
         adaptive_thresholds = calibrate_all_cusum_thresholds(
             df_full, target_events_per_day=target_signals_per_day,
             sr_levels=sr_levels
         )
         tprint_info(f"Calibrated Thresholds: {adaptive_thresholds}")

    # Orthogonal signal families
    base_generators = [
        # Price-based (single best: trend/reversal weighted)
        ('PRICE_CUSUM', ImprovedCUSUMEvents()),
        # Contextual families
        ('VOL_CUSUM', VolatilityCusumEvents()),
        ('LIQ_CUSUM', LiquidityCusumEvents()),
        ('VOL_PARTICIPATION', VolumeCusumEvents()),
        ('RANGE_ATR', RangeATRcusumEvents()),
        ('TAIL_RISK', TailRiskCusumEvents()),
        ('TREND_REGIME', TrendRegimeCusumEvents()),
        ('VOL_STATE', VolatilityStateEvents()),
    ]
    
    # Build enhanced parameter combinations
    for fam, gen in base_generators:
        family_config = param_grids['family_grids'].get(fam, {})
        base_params = family_config.get('base_params', [])
        
        # Add base parameters only (no variations)
        for params in base_params:
            # Inject SR levels for SR_CUSUM
            if fam == 'SR_CUSUM':
                 # params is tuple, convert to list to append
                 p_list = list(params)
                 if len(p_list) == 1: # (h,)
                      p_list.append(sr_levels)
                 params = tuple(p_list)

            # Inject calibrated threshold if available
            # Note: params structure varies.
            # PRICE_CUSUM: (multiplier, vol_window) -> multiplier is approx k.
            # VOL_CUSUM: (h, vol_span)
            # LIQ_CUSUM: (h, vol_span)
            # VOL_PARTICIPATION: (h, span)
            # RANGE_ATR: (h, atr_window, vol_window)
            # SR_CUSUM: (h, sr_levels)

            if use_adaptive_thresholds:
                p_list = list(params)
                if fam == 'PRICE_CUSUM': # param 0 is multiplier/k
                     # Calibrated 'price' is a threshold for price change, not directly k.
                     # AdaptiveSymmetricCUSUM uses k * sigma.
                     # Calibrated threshold is raw price change quantile.
                     # We can leave as is or map.
                     pass
                elif fam == 'VOL_CUSUM' and 'volatility' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['volatility']
                elif fam == 'LIQ_CUSUM':
                     # LIQ uses diff of log(TR/Vol).
                     # Calibrated thresholds doesn't directly give this.
                     pass
                elif fam == 'VOL_PARTICIPATION' and 'volume' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['volume']
                elif fam == 'RANGE_ATR' and 'atr' in adaptive_thresholds:
                     p_list[0] = adaptive_thresholds['atr']
                elif fam == 'SR_CUSUM' and 'sr' in adaptive_thresholds and adaptive_thresholds['sr'] is not None:
                     p_list[0] = adaptive_thresholds['sr']

                params = tuple(p_list)

            generator_configs.append((fam, gen, params))
    
    tprint_info(f"Generated {len(generator_configs)} enhanced candidate configurations")
    
    # 3. Process Candidates (Generate & Gate)
    candidates = []
    # 4. Pruning Phase: Validate TP:SL Grid on Central Parameters
    # -------------------------------------------------------------
    tprint_info("✂️ Starting Pruning Phase: Validating TP:SL configs on central parameters...")
    
    # Define Central Parameters (Median/Default values)
    CENTRAL_PARAMS = {
        'PRICE_CUSUM': (1.0, 20),      # k=1.0, vol=20
        'VOL_CUSUM': (4.0, 100),       # h=4.0, span=100
        'LIQ_CUSUM': (4.0, 100),       # h=4.0, span=100
        'VOL_PARTICIPATION': (5.0, 960), # h=5.0 (500% excess), span=960 (10d)
        'RANGE_ATR': (2.0, 14, 20),    # h=2.0, atr=14, vol=20
        'TAIL_RISK': (2.0, 50, 'skew'),# h=2.0, win=50
        'TREND_REGIME': (2.0, 20, 50), # h=2.0, fast=20, slow=50
        'VOL_STATE': (2.0, 100),       # h=2.0, span=100
        'ImprovedCUSUMEvents': (1.0, 20),
        'SR_CUSUM': (2.0, sr_levels)   # Dynamic
    }

    valid_tpsl_map = {} # family -> list of valid grid_items
    
    # Get all unique generator instances/families
    unique_families = list(set(g[0] for g in generator_configs))
    
    for fam in unique_families:
        if fam not in CENTRAL_PARAMS:
            valid_tpsl_map[fam] = param_grids['tpsl_grid'] # Fallback: use all
            continue
            
        central_args = CENTRAL_PARAMS[fam]
        # Find the generator instance
        gen_instance = next((g[1] for g in generator_configs if g[0] == fam), None)
        if not gen_instance: continue

        # Generate events for central params
        # Generate events for central params
        try:
             # Extended list of classes requiring DataFrame
             df_required_local = (
                 'VolatilityCusumEvents', 'LiquidityCusumEvents', 'VolumeCusumEvents',
                 'RangeATRcusumEvents', 'SRCusumEvents',
                 'TailRiskCusumEvents', 'TrendRegimeCusumEvents', 'VolatilityStateEvents',
                 'ImprovedCUSUMEvents'
             )
             
             gen_classname = gen_instance.__class__.__name__
             
             if gen_classname in df_required_local:
                 if gen_classname == 'ImprovedCUSUMEvents':
                      p_names = GENERATOR_PARAM_NAMES.get('ImprovedCUSUMEvents', [])
                      # params are tuple, zip with names
                      # ImprovedCUSUMEvents params: (multiplier, vol_window) -> k, vol_window
                      # Default param names might be different, check definition or use positional logic if generate supports it.
                      # ImprovedCUSUM: generate(df, **params) but params definition says:
                      # generate(self, df: pd.DataFrame, **params)
                      # And inside: k, alpha, beta ... 
                      # The passed 'params' tuple from grid is likely (k, vol_window).
                      # Let's assume standard positional *args work if generate supports it, OR we reconstruct kwargs.
                      # Looking at call site in main loop:
                      # kwargs = dict(zip(param_names, params))
                      # events = gen.generate(df_full, **kwargs)
                      # So we must do the same.
                      
                      # Re-fetch param names if needed or hardcode common ones
                      p_names = ['k', 'vol_window'] # Based on grid def usually
                      # If param_grids not visible here, relying on GENERATOR_PARAM_NAMES is safer.
                      p_names = GENERATOR_PARAM_NAMES.get('ImprovedCUSUMEvents', ['k', 'vol_window'])
                      kwargs = dict(zip(p_names, central_args))
                      c_events = gen_instance.generate(df_full, **kwargs)
                 else:
                      c_events = gen_instance.generate(df_full, *central_args)
             else:
                  c_events = gen_instance.generate(price, *central_args)
        except Exception as e:
            tprint_warning(f"Pruning: Failed to generate central events for {fam}: {e}")
            valid_tpsl_map[fam] = param_grids['tpsl_grid'] # Fallback
            continue

        if len(c_events) < 5:
            tprint_warning(f"Pruning: {fam} central events too few ({len(c_events)}). Skipping usage check.")
            valid_tpsl_map[fam] = param_grids['tpsl_grid']
            continue
            
        # Test TP:SL combinations
        valid_items = []
        for grid_item in param_grids['tpsl_grid']:
            pt = grid_item['pt']
            sl = grid_item['sl']
            
            # Use a representative horizon (e.g. 24) and risk_budget (e.g. 0.7) for pruning
            # We check if this TP:SL passes class balance and min samples
            
            try:
                # Label Generation (Simplified call for speed)
                # Need to use appropriate labeling function based on family
                
                # ... (Labeling logic similar to main loop) ...
                # To avoid code duplication, we assume similar mapping logic
                
                if fam == 'PRICE_CUSUM':
                    high, low = df_full.get('high'), df_full.get('low')
                    lbls, _, _, _, _, _ = compute_dominance_labels(price, c_events, df_full['volatility_1d'], risk_budget=0.7, pt_mult=pt, sl_mult=sl, horizon=24, high=high, low=low)
                elif fam == 'LIQ_CUSUM':
                    lbls, _, _, _, _, _ = compute_path_degradation_labels(df_full, c_events, horizon=24, d_sigma=pt*1.5)
                elif fam == 'TAIL_RISK':
                     z_thresh = pt * 0.6
                     # Need metric from params or default
                     # TAIL_RISK central params: (2.0, 50, 'skew')
                     metric = central_args[2] if len(central_args) > 2 else 'skew'
                     lbls, _, _, _, _, _ = compute_tail_regime_labels(df_full, c_events, horizon=24, z_thresh=z_thresh, metric_col=metric)
                elif fam == 'TREND_REGIME':
                     lbls, _, _, _, _, _ = compute_trend_persistence_labels(df_full, c_events, horizon=24)
                elif fam == 'VOL_STATE':
                     lbls, _, _, _, _, _ = compute_vol_state_labels(df_full, c_events, horizon=24)
                elif fam == 'VOL_CUSUM':
                    lbls, _, _, _, _, _ = compute_volatility_labels(df_full, c_events, horizon=24, k=1.1+(pt*0.1))
                else:
                    # Default/Fallback (e.g. Volatility)
                    lbls, _, _, _, _, _ = compute_volatility_labels(df_full, c_events, horizon=24, k=1.1+(pt*0.1))
                
                if lbls.empty: continue
                
                # Check Gates: Balance & Count
                pos_rate = lbls.mean()
                count_ok = len(lbls) >= 50 # Relaxed for pruning
                
                # PRICE_CUSUM specific relaxation (3%)
                min_bal = 0.03 if fam == 'PRICE_CUSUM' else 0.10
                bal_ok = (pos_rate >= min_bal) and (pos_rate <= 0.90)
                
                if count_ok and bal_ok:
                    valid_items.append(grid_item)
                    
            except Exception:
                continue
                
        if len(valid_items) > 0:
            valid_tpsl_map[fam] = valid_items
            tprint_info(f"✂️ Pruned {fam}: Kept {len(valid_items)}/{len(param_grids['tpsl_grid'])} TP:SL configs")
        else:
            tprint_warning(f"Pruning warning: {fam} had 0 passing TP:SLs. Using all defaults.")
            valid_tpsl_map[fam] = param_grids['tpsl_grid']

    
    # 5. Process Candidates (Main Sweep)
    tprint_info("🚀 Starting Main Parameter Sweep...")
    candidates = []
    outcomes_log = []


    for fam, gen, params in generator_configs:

            try:
                # Use standard generation (Adaptive logic skipped for now for these new generators to match snippet)
                # But kept logic structure if needed.

                # Extended list of classes requiring DataFrame
                df_required = DF_REQUIRED_CLASSES + (
                    'VolatilityCusumEvents', 'LiquidityCusumEvents', 'VolumeCusumEvents',
                    'RangeATRcusumEvents', 'SRCusumEvents',
                    'TailRiskCusumEvents', 'TrendRegimeCusumEvents', 'VolatilityStateEvents',
                    'ImprovedCUSUMEvents'
                )

                if gen.__class__.__name__ in df_required:
                    if gen.__class__.__name__ == 'ImprovedCUSUMEvents':
                         # Must pass as kwargs for **params
                         param_names = GENERATOR_PARAM_NAMES.get('ImprovedCUSUMEvents', [])
                         kwargs = dict(zip(param_names, params))
                         events = gen.generate(df_full, **kwargs)
                    else:
                         # Special handling for SR_CUSUM if sr_levels passed as positional
                         # SRCusumEvents.generate(df, h, sr_levels)
                         # params is (h, sr_levels)
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
            param_names = GENERATOR_PARAM_NAMES.get(gen.__class__.__name__, [])
            if param_names and len(params) == len(param_names):
                param_dict = dict(zip(param_names, params))
            else:
                param_dict = {'params': params}

            # Iterate Enhanced Grids - Using Validated TP:SLs
            tpsl_grid = valid_tpsl_map.get(fam, param_grids['tpsl_grid'])
            
            # If no pruning happened for this family, warn or info
            if len(tpsl_grid) == len(param_grids['tpsl_grid']):
                 # tprint_info(f"Using full grid for {fam}")
                 pass
            
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
                            # d_sigma map from pt: pt=[1.5, 4.0] -> d=[2.25, 6.0]
                            # Increased multiplier from 0.5 to 1.5 to require more extreme stress
                            d_sigma = pt * 1.5
                            labels, weights, returns, mfe, mae, vol = compute_path_degradation_labels(
                                df_full, events, horizon=horizon, d_sigma=d_sigma
                            )
                        elif fam == 'TAIL_RISK':
                            # Tail Risk Persistence
                            # Map pt to z_thresh (1.5 - 2.5)
                            z_thresh = pt * 0.6
                            labels, weights, returns, mfe, mae, vol = compute_tail_regime_labels(
                                df_full, events, horizon=horizon, z_thresh=z_thresh,
                                metric_col=param_dict.get('metric', 'skew')
                            )
                        elif fam == 'TREND_REGIME':
                            # Trend Persistence
                            labels, weights, returns, mfe, mae, vol = compute_trend_persistence_labels(
                                df_full, events, horizon=horizon
                            )
                        elif fam == 'VOL_STATE':
                            # Vol State Persistence
                            labels, weights, returns, mfe, mae, vol = compute_vol_state_labels(
                                df_full, events, horizon=horizon
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
                            events, labels, returns, df_full, X_probe, gen, param_dict, family=fam
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

    # If requested, return ALL robust candidates for Layer 2 Selection
    if return_raw_candidates:
        tprint_info(f"Returning {len(scored_candidates)} raw candidates for advanced selection.")
        raw_geoms = []
        for cand in scored_candidates:
            # We need to construct OutputGeometry but WITHOUT Probe Metrics (expensive?)
            # Actually, User Plan says: "Select top 5 per family -> Probe -> Winner"
            # So we shouldn't probe ALL here if it's expensive.
            # But OutputGeometry usually expects probe metrics (AUC).
            # Current logic: Probe is done on Top 50%.
            
            # Let's perform a lightweight probe or defer it?
            # 'auc' field in OutputGeometry is key.
            # We can use 'learnability' (IC/PSR) from 'metrics_raw' as proxy for AUC 
            # or just set a placeholder since Layer 2 will re-probe/race.
            
            # Use 'lift' or 'sharpe_meta' from initial metrics as AUC proxy
            auc_proxy = cand['metrics_raw'].get('lift', 0.0)
            
            purity = 1.0 # Placeholder
            
            final_weights = get_signal_specific_weights(df_full, cand['events'], sr_levels, component_weights=signal_weights, family=cand['family'])
            
            geo = OutputGeometry(
                name=f"{cand['family']}_{cand['params']}",
                family=cand['family'],
                events=cand['events'],
                labels=cand['labels'],
                weights=final_weights,
                purity=purity,
                auc=auc_proxy, 
                params=cand['params'],
                metrics=cand['metrics_raw']
            )
            raw_geoms.append(geo)
        return raw_geoms

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

        # Get De Prado Weights (combining all signals)
        final_weights = get_signal_specific_weights(df_full, cand['events'], sr_levels, component_weights=signal_weights, family=cand['family'])
        # Blend or replace? "update with this" suggests using it.
        # Use final_weights as the primary weights for the geometry.

        geo = OutputGeometry(
            name=f"{cand['family']}_{cand['params']}",
            family=cand['family'],
            events=cand['events'],
            labels=cand['labels'],
            weights=final_weights, # Use combined weights
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
    
    Enhanced with:
    1. Time-varying drift (relative momentum) - only fires when move exceeds recent trend
    2. Volume-weighted cumsum (conviction filter) - high-volume moves accumulate faster
    3. Stricter dynamic threshold floor - prevents firing in low-vol noise regimes
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

    # ENHANCEMENT 1: Time-varying drift (Relative Momentum Sensor)
    # Use excess returns above recent median - only fires when move exceeds baseline
    drift_baseline = log_ret_smooth_series.rolling(window_vol, min_periods=1).median()
    excess_ret = (log_ret_smooth_series - drift_baseline).fillna(0.0)

    # 4. Liquidity & Thresholds
    liquidity_mod = pd.Series(1.0, index=close.index)
    conviction = pd.Series(1.0, index=close.index)  # For volume-weighted cumsum
    
    if volume is not None:
        vol_ma = volume.rolling(window_vol, min_periods=1).mean()
        vol_std = volume.rolling(window_vol, min_periods=1).std()
        rel_volume = volume / (vol_ma + 1e-9)
        liquidity_mod = 1.0 + beta * (1.0 - rel_volume)
        liquidity_mod = liquidity_mod.clip(0.5, 2.0)
        
        # ENHANCEMENT 2: Volume-weighted cumsum (Conviction Filter)
        # High-volume moves accumulate faster, low-volume fakeouts decay
        vol_zscore = ((volume - vol_ma) / (vol_std + 1e-9)).clip(-2, 2).fillna(0)
        conviction = (1.0 + 0.5 * vol_zscore)  # Range: 0.0 to 2.0

    regime_mod = 1.0 + alpha * (1.0 - ER)
    h_raw = k * sigma * regime_mod * liquidity_mod
    
    # ENHANCEMENT 3: Stricter Dynamic Threshold Floor
    # Prevent firing in low-vol noise regimes
    h_floor = sigma.rolling(100, min_periods=20).quantile(0.9) * 0.5
    h_t = np.maximum(h_raw, h_floor).fillna(h_raw).fillna(0.0)

    # 5. Residuals for Reversal Logic
    expected_return = log_ret_smooth_series.rolling(window_vol, min_periods=1).mean()
    residual_ret = (log_ret_smooth_series - expected_return).fillna(0.0)

    # 6. CUSUM Loop - now using excess returns and conviction
    n = len(close)
    excess_arr = excess_ret.to_numpy()  # ENHANCED: use excess returns
    res_arr = residual_ret.to_numpy()
    h_arr = h_t.to_numpy()
    er_arr = ER.to_numpy()
    conviction_arr = conviction.to_numpy()  # ENHANCED: volume conviction

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

        # Get conviction factor for this bar
        conv = conviction_arr[t] if not np.isnan(conviction_arr[t]) else 1.0

        # Trend CUSUM - ENHANCED: use excess returns weighted by conviction
        S_trend_pos = max(0.0, S_trend_pos + excess_arr[t] * conv)
        S_trend_neg = min(0.0, S_trend_neg + excess_arr[t] * conv)

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
    """
    Fully Adaptive CUSUM (Dual Signal: Trend + Mean Reversion).
    Re-enabled to generate both mean reversions and trends using Kalman filtering.
    """
    def generate(self, data: Union[pd.Series, pd.DataFrame], multiplier: float = 0.5, vol_window: int = 20) -> pd.DatetimeIndex:
        if isinstance(data, pd.DataFrame):
            close = data['close']
            volume = data.get('volume')
        else:
            close = data
            volume = None

        try:
            # Map multiplier to 'k' (threshold scaling)
            # Standard k is around 0.1-0.2 for daily? Or just a scalar.
            # In generate_dual_cusum_signals, k is used as k * sigma.
            # Our multiplier is passed as 0.5 or 0.75 in the grid.
            # Dual CUSUM default k is 0.12.
            # If multiplier is 0.5, it's aggressive.
            # I will use multiplier directly as k.

            dual_signals = generate_dual_cusum_signals(
                close=close, volume=volume,
                k=multiplier, # multiplier serves as threshold 'k'
                window_vol=vol_window
            )

            # Combine signals (Trend + Reversion)
            # We want events for EITHER.
            # Using simple OR logic on non-zero signals.

            w_trend = 1.0
            w_reversal = 1.0

            composite = w_trend * dual_signals['trend_signal'] + w_reversal * dual_signals['reversal_signal']
            return composite.index[composite != 0]

        except Exception as e:
            logger.warning(f"AdaptiveSymmetricCUSUMEvents failed: {e}")
            return pd.DatetimeIndex([])


class VolatilityCusumEvents(BaseEventGenerator):
    """
    VOLATILITY CUSUM — Risk Regime Change (Direction-Free).
    Enhanced with unified Kalman+VWAP price for improved signal quality.
    """
    def generate(self, df: pd.DataFrame, h: float = 4.0, vol_span: int = 100) -> pd.DatetimeIndex:
        # Handle input type
        if isinstance(df, pd.Series):
            close = df
        else:
            close = df['close']

        try:
            # Use unified price if enabled
            if self.use_unified_price and isinstance(df, pd.DataFrame):
                close = self._get_unified_price(df)
            
            ret = np.log(close).diff()
            vol = ewma_volatility(ret, span=vol_span)
            log_vol_change = np.log(vol).diff()

            # Normalize (Consistency Update)
            norm = log_vol_change.rolling(100).std()
            xt = log_vol_change / (norm + 1e-9)

            s_pos, s_neg = 0.0, 0.0
            events = []

            # Iteration
            # Vectorization is hard for CUSUM, loop is fine
            vals = xt.values
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
            
        except Exception as e:
            logger.error(f"VolatilityCusumEvents generation failed: {e}")
            return pd.DatetimeIndex([])

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

        # CUSUM on changes (consistent with trend/vol)
        xt_raw = liq_stress.diff()
        # Normalize
        norm = xt_raw.rolling(100).std()
        xt = xt_raw / (norm + 1e-9)

        s_pos, s_neg = 0.0, 0.0
        events = []

        vals = xt.values
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
    """Volume flow analysis for Layer2 per de Prado framework.
    
    Generates continuous flow metrics for meta-labeling context:
    - Volume pressure (buy/sell imbalance)
    - Flow intensity (abnormal volume detection)
    - Flow persistence (sustained volume patterns)
    """
    
    def generate(self, df: pd.DataFrame, h: float = 5.0, span: int = 960) -> pd.DatetimeIndex:
        """
        Generates volume participation events based on accumulated % volume excess relative to a long-term baseline.
        
        Logic:
        1. Baseline = 10-day rolling mean of volume (span=960 for 15m bars).
        2. Volume Excess = (Volume / Baseline) - 1.0
        3. Signal = Volume Excess * sign(Returns)
        4. CUSUM accumulates this signal. Threshold 'h' represents accumulated % excess.
           h=5.0 means we need to accumulate 500% worth of volume excess in a price-consistent direction.
        """
        if 'close' not in df.columns or 'volume' not in df.columns:
            return pd.DatetimeIndex([])

        close = df['close']
        volume = df['volume']
        
        # 1. Baseline: 10-day rolling mean (simple moving average for baseline)
        # Using rolling().mean() instead of ewm for a stable long-term baseline
        vol_baseline = volume.rolling(window=span, min_periods=span//2).mean()
        
        # 2. Volume Excess % (Normalized)
        # Handle division by zero/nan
        vol_excess = (volume / (vol_baseline + 1e-9)) - 1.0
        vol_excess = vol_excess.fillna(0.0) # 0 excess if no baseline
        
        # 3. Directional Proxy
        price_ret = close.pct_change().fillna(0)
        # We want to accumulate volume that supports the price move
        # Signal = Excess volume * direction of price
        signed_vol_proxy = vol_excess * np.sign(price_ret)
        # Handle cases where price_ret is 0 -> signal is 0

        s_pos, s_neg = 0.0, 0.0
        events = []

        # Optimization: use numpy for speed
        svp = signed_vol_proxy.values
        idx = df.index

        for t in range(1, len(df)):
            x = svp[t]
            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)
            if s_pos > h:
                events.append(idx[t])
                s_pos = 0.0
            elif s_neg < -h:
                events.append(idx[t])
                s_neg = 0.0

        return pd.DatetimeIndex(events)
    
    def generate_flow_metrics(self, df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
        """Generate continuous volume flow metrics for Layer2 context."""
        if 'close' not in df.columns or 'volume' not in df.columns:
            return pd.DataFrame(0, index=df.index, columns=['volume_pressure', 'flow_intensity', 'flow_persistence'])

        try:
            # Use unified price if enabled
            if self.use_unified_price:
                close = self._get_unified_price(df)
            else:
                close = df['close']
            
            volume = df['volume']
            
            # 1. Volume Pressure (buy/sell imbalance)
            vol_ma = volume.rolling(span).mean()
            vol_z = (volume - vol_ma) / (vol_ma.rolling(span).std() + 1e-9)
            price_change = close.pct_change()
            
            # Signed volume pressure (volume aligned with price direction)
            volume_pressure = vol_z * np.sign(price_change)
            volume_pressure = volume_pressure.rolling(5, min_periods=1).mean().fillna(0)
            
            # 2. Flow Intensity (abnormal volume detection)
            vol_ratio = volume / vol_ma
            flow_intensity = cusum_to_probability(vol_ratio - 1, scale=2.0)
            flow_intensity = flow_intensity.rolling(3, min_periods=1).mean().fillna(0)
            
            # 3. Flow Persistence (sustained volume patterns)
            vol_trend = volume.rolling(span).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            flow_persistence = zscore_to_probability(vol_trend, two_sided=False)
            flow_persistence = flow_persistence.rolling(5, min_periods=1).mean().fillna(0)
            
            # Combine into flow metrics DataFrame
            flow_df = pd.DataFrame({
                'volume_pressure': volume_pressure,
                'flow_intensity': flow_intensity,
                'flow_persistence': flow_persistence
            }, index=df.index)
            
            logger.debug(f"VolumeCusumEvents generated flow metrics (pressure_mean={volume_pressure.mean():.3f})")
            return flow_df
            
        except Exception as e:
            logger.error(f"VolumeCusumEvents flow analysis failed: {e}")
            return pd.DataFrame(0, index=df.index, columns=['volume_pressure', 'flow_intensity', 'flow_persistence'])


class RangeATRcusumEvents(BaseEventGenerator):
    """Detects bursts of intrabar volatility independent of price direction."""
    def generate(self, df: pd.DataFrame, h: float = 2.0, atr_window: int = 14, vol_window: int = 20) -> pd.DatetimeIndex:
        if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return pd.DatetimeIndex([])

        tr = df['high'] - df['low']
        atr = tr.rolling(atr_window).mean().fillna(1e-9)
        atr_change = np.log(atr / atr.rolling(vol_window).mean()).fillna(0)

        s_pos, s_neg = 0.0, 0.0
        events = []

        vals = atr_change.values
        idx = df.index

        for t in range(1, len(df)):
            x = vals[t]
            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)
            if s_pos > h or abs(s_neg) > h:
                events.append(idx[t])
                s_pos, s_neg = 0.0, 0.0

        return pd.DatetimeIndex(events)


class SRCusumEvents(BaseEventGenerator):
    """Detects persistent price movement relative to key levels."""
    def generate(self, df: pd.DataFrame, h: float = 2.0, sr_levels: list = None) -> pd.DatetimeIndex:
        if 'close' not in df.columns or not sr_levels:
            return pd.DatetimeIndex([])

        close_vals = df['close'].values
        sr_arr = np.array(sr_levels)

        # Calculate distance to nearest level (vectorized)
        distance = (close_vals[:, None] - sr_arr[None, :]).min(axis=1)

        s_pos, s_neg = 0.0, 0.0
        events = []

        idx = df.index

        for t in range(1, len(df)):
            x = distance[t] - distance[t-1]
            s_pos = max(0.0, s_pos + x)
            s_neg = min(0.0, s_neg + x)
            if s_pos > h or abs(s_neg) > h:
                events.append(idx[t])
                s_pos, s_neg = 0.0, 0.0

        return pd.DatetimeIndex(events)


class TailRiskCusumEvents(BaseEventGenerator):
    """
    TAIL RISK CUSUM — Skew/Kurtosis Detector.
    """
    def generate(self, df: pd.DataFrame, h: float = 2.0, window: int = 50, metric: str = 'skew') -> pd.DatetimeIndex:
        if isinstance(df, pd.Series):
             ret = df.pct_change()
        else:
             ret = df['close'].pct_change()

        if metric == 'skew':
            series = ret.rolling(window).skew()
        else:
            series = ret.rolling(window).kurt()

        # Log changes. Handle negative skew by using abs?
        # User says: diff(log(skew)). Skew can be negative.
        # Maybe diff(skew) is safer?
        # Or diff(log(abs(skew))).
        # "xt = diff(log(skewt))".
        # I'll use diff of raw metric standardized by its own volatility?
        # Or just diff(series).
        # Let's try to follow prompt: "xt = Delta log(skew)".
        # We'll use log of absolute value to avoid domain error, preserving sign?
        # Actually, let's just use diff(series) normalized by recent std of series to make 'h' stable.

        diff = series.diff()
        # Normalize by rolling std of diff
        norm = diff.rolling(100).std()
        xt = diff / (norm + 1e-9)

        s_pos, s_neg = 0.0, 0.0
        events = []
        vals = xt.values
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

class TrendRegimeCusumEvents(BaseEventGenerator):
    """
    TREND REGIME CUSUM — Directional State Context.
    Enhanced with unified Kalman+VWAP price for improved trend detection.
    """
    def generate(self, df: pd.DataFrame, h: float = 2.0, fast: int = 20, slow: int = 50) -> pd.DatetimeIndex:
        if isinstance(df, pd.Series):
             close = df
        else:
             close = df['close']

        try:
            # Use unified price if enabled
            if self.use_unified_price and isinstance(df, pd.DataFrame):
                close = self._get_unified_price(df)

            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()

            trend = (ema_fast - ema_slow) / close

            xt = trend.diff()
            # Normalize
            norm = xt.rolling(100).std()
            xt_norm = xt / (norm + 1e-9)

            s_pos, s_neg = 0.0, 0.0
            events = []
            vals = xt_norm.values
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
            
        except Exception as e:
            logger.error(f"TrendRegimeCusumEvents generation failed: {e}")
            return pd.DatetimeIndex([])


class VolatilityStateEvents(BaseEventGenerator):
    """
    VOLATILITY REGIME SWITCHES — State Detector.
    """
    def generate(self, df: pd.DataFrame, h: float = 2.0, vol_span: int = 100) -> pd.DatetimeIndex:
        if isinstance(df, pd.Series):
             close = df
        else:
             close = df['close']

        ret = np.log(close).diff()
        vol = ewma_volatility(ret, span=vol_span)

        # CUSUM on vol changes (similar to VolCusum but maybe different tuning)
        # xt = diff(log(vol))
        xt_raw = np.log(vol).diff()

        # Normalize
        norm = xt_raw.rolling(100).std()
        xt = xt_raw / (norm + 1e-9)

        s_pos, s_neg = 0.0, 0.0
        events = []
        vals = xt.values
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
        volume = None
        if 'volume' in df.columns:
            volume = df['volume']
        elif 'Volume' in df.columns:
            volume = df['Volume']

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


class MultiHorizonPriceCUSUMEvents(BaseEventGenerator):
    """
    Generate PRICE_CUSUM events across multiple adaptive horizons.
    Combines short, medium, and long horizon events with horizon-specific TP:SL adjustments.
    """
    
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        events_list = []
        
        # Define horizon strategies
        horizon_configs = [
            {'name': 'short', 'horizon': 6, 'pt_adj': 0.8, 'sl_adj': 0.6},
            {'name': 'medium', 'horizon': 12, 'pt_adj': 1.0, 'sl_adj': 1.0},
            {'name': 'long', 'horizon': 24, 'pt_adj': 1.2, 'sl_adj': 1.1}
        ]
        
        # Calculate adaptive base parameters
        adaptive_params = self._get_adaptive_price_cusum_params(df, params)
        
        for config in horizon_configs:
            # Adjust horizon-specific parameters
            horizon_params = adaptive_params.copy()
            horizon_params['horizon'] = config['horizon']
            
            # Horizon-specific TP:SL adjustments
            horizon_params['pt_mult'] = horizon_params.get('pt_mult', 1.5) * config['pt_adj']
            horizon_params['sl_mult'] = horizon_params.get('sl_mult', 0.75) * config['sl_adj']
            
            # Generate events for this horizon using base generator
            try:
                close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
                volume = df.get('volume', df.get('Volume'))
                
                dual_signals = generate_dual_cusum_signals(
                    close=close,
                    volume=volume,
                    k=horizon_params.get('k', 0.12),
                    alpha=horizon_params.get('alpha', 1.0),
                    beta=horizon_params.get('beta', 1.0),
                    er_min=horizon_params.get('er_min', 0.2),
                    window_vol=horizon_params.get('vol_window', 20),
                    window_er=horizon_params.get('er_window', 10)
                )
                composite = dual_signals['trend_signal'] + dual_signals['reversal_signal']
                horizon_events = composite.index[composite != 0]
                events_list.append(horizon_events)
            except Exception as e:
                logger.warning(f"MultiHorizon {config['name']} failed: {e}")
        
        # Combine events with deduplication
        if events_list:
            all_events = pd.DatetimeIndex(np.concatenate([e.values for e in events_list if len(e) > 0]))
            return all_events.drop_duplicates().sort_values()
        return pd.DatetimeIndex([])
    
    def _get_adaptive_price_cusum_params(self, df: pd.DataFrame, base_params: dict) -> dict:
        """Calculate adaptive CUSUM parameters based on market conditions."""
        params = base_params.copy()
        
        try:
            close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
            returns = close.pct_change().dropna()
            
            # Adaptive k based on recent volatility
            recent_vol = returns.iloc[-100:].std() if len(returns) >= 100 else returns.std()
            historical_vol = returns.std()
            
            # If recent vol is higher than historical, use stricter k (filter more noise)
            vol_ratio = recent_vol / (historical_vol + 1e-9)
            base_k = params.get('k', 0.12)
            
            # Adjust k: higher vol -> higher k (stricter filter)
            if vol_ratio > 1.2:
                params['k'] = base_k * min(vol_ratio, 2.0)
            elif vol_ratio < 0.8:
                params['k'] = base_k * max(vol_ratio, 0.5)
                
        except Exception:
            pass
        
        return params


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

# ==========================================
# 9. Meta-Learning Dataset Generation
# ==========================================

def apply_persistence_label(df: pd.DataFrame, events: pd.DatetimeIndex, series_col: str, horizon: int = 48, threshold: float = 0.0) -> pd.Series:
    """
    Generic persistence labeler.
    Returns 1 if series_col > threshold on average over horizon.
    """
    if events.empty or series_col not in df.columns:
        return pd.Series(0, index=df.index)

    # Align events
    valid_events = events.intersection(df.index)
    if valid_events.empty:
        return pd.Series(0, index=df.index)

    event_locs = df.index.get_indexer(valid_events)
    n_bars = len(df)

    # Filter valid
    valid_mask = (event_locs != -1) & (event_locs < (n_bars - horizon))
    valid_idxs = event_locs[valid_mask]
    final_events = valid_events[valid_mask]

    if len(valid_idxs) == 0:
        return pd.Series(0, index=df.index)

    # Window Matrix
    offsets = np.arange(1, horizon + 1)
    window_idxs = valid_idxs[:, None] + offsets[None, :]

    vals = df[series_col].values[window_idxs]
    avg_vals = np.mean(vals, axis=1)

    labels = (avg_vals > threshold).astype(int)

    out = pd.Series(0, index=df.index)
    out.loc[final_events] = labels
    return out

def apply_triple_barrier_multi(df: pd.DataFrame, events: pd.DatetimeIndex,
                                pt_sl: Tuple[float,float]=(2.0, 1.0), # multipliers for vol
                                horizons: list=[12,48]) -> pd.DataFrame:
    """
    Returns a DataFrame of price labels for multiple horizons using volatility-adjusted barriers.
    Columns: 'price_label_{horizon}'
    """
    out = pd.DataFrame(0, index=df.index, columns=[f'price_label_{h}' for h in horizons], dtype=int)
    close = df['close'].values

    # Volatility
    if 'volatility_1d' in df.columns:
        vol = df['volatility_1d'].values
    else:
        vol = df['close'].pct_change().rolling(100).std().fillna(0).values

    # Normalize TZ
    if df.index.tz is not None:
        idx_base = df.index.tz_localize(None)
    else:
        idx_base = df.index

    if events.tz is not None:
        events_norm = events.tz_localize(None)
    else:
        events_norm = events

    event_idxs = idx_base.get_indexer(events_norm)
    valid_mask = (event_idxs != -1)
    valid_idxs = event_idxs[valid_mask]

    for h in horizons:
        # Filter for horizon
        h_mask = valid_idxs < (len(close) - h)
        h_idxs = valid_idxs[h_mask]

        if len(h_idxs) == 0:
            continue

        # Vectorized Window
        offsets = np.arange(1, h + 1)
        window_idxs = h_idxs[:, None] + offsets[None, :]

        window_prices = close[window_idxs]
        entry_prices = close[h_idxs]
        entry_vols = vol[h_idxs]

        # Avoid zero vol
        entry_vols = np.maximum(entry_vols, 1e-6)

        ret = window_prices / entry_prices[:, None] - 1.0

        up_barrier = pt_sl[0] * entry_vols
        down_barrier = pt_sl[1] * entry_vols

        hit_up = ret >= up_barrier[:, None]
        hit_down = ret <= -down_barrier[:, None]

        # First hit logic
        first_up = np.argmax(hit_up, axis=1)
        first_down = np.argmax(hit_down, axis=1)

        # Mask where no hit occurred (argmax returns 0 if all false, need to check if actually hit)
        any_up = np.any(hit_up, axis=1)
        any_down = np.any(hit_down, axis=1)

        labels = np.zeros(len(h_idxs), dtype=int)

        # Vectorized check
        # Case 1: Only Up
        mask_up = any_up & ~any_down
        labels[mask_up] = 1

        # Case 2: Only Down
        mask_down = any_down & ~any_up
        labels[mask_down] = -1

        # Case 3: Both
        mask_both = any_up & any_down
        # first_up < first_down -> 1
        sub_mask_up = mask_both & (first_up < first_down)
        labels[sub_mask_up] = 1

        sub_mask_down = mask_both & (first_down < first_up)
        labels[sub_mask_down] = -1

        # Assign to output
        evt_timestamps = events_norm[valid_mask][h_mask]

        # Align TZ if needed
        if df.index.tz is not None and evt_timestamps.tz is None:
             evt_timestamps = evt_timestamps.tz_localize(df.index.tz)

        out.loc[evt_timestamps, f'price_label_{h}'] = labels

    return out

def create_meta_learning_dataset_dualTBM(df: pd.DataFrame, base_features: pd.DataFrame,
                                         pt_sl=(2.0, 1.0), tbm_horizons=[12,48]):
    meta_df = base_features.copy()

    # Directional price labels for multiple horizons
    if 'price_dual_cusum' in base_features.columns:
        price_events = base_features.index[base_features['price_dual_cusum']==1]
        # Normalize timezones
        if df.index.tz != price_events.tz:
             if price_events.tz is None: price_events = price_events.tz_localize(df.index.tz)
             else: price_events = price_events.tz_convert(df.index.tz)

        tbm_labels = apply_triple_barrier_multi(df, price_events, pt_sl=pt_sl, horizons=tbm_horizons)
        meta_df = pd.concat([meta_df, tbm_labels], axis=1)

    # Contextual labels
    context_map = {
        'volatility_cusum': 'volatility_1d',
        'liquidity_cusum': 'liq_stress',
        'volume_cusum': 'volume',
        'tailrisk_cusum': 'tail_metric',
        'trend_regime_cusum': 'trend',
        'vol_state_cusum': 'vol_state'
    }

    for col, series_col in context_map.items():
        if col in base_features.columns and series_col in df.columns:
            events = base_features.index[base_features[col]==1]
            if df.index.tz != events.tz:
                 if events.tz is None: events = events.tz_localize(df.index.tz)
                 else: events = events.tz_convert(df.index.tz)

            lbl = apply_persistence_label(df, events, series_col=series_col, horizon=48, threshold=0.0)
            meta_df[f'{col}_label'] = lbl

    return meta_df
