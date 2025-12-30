import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from itertools import combinations, product
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import entropy as shannon_entropy
from scipy.special import expit
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import List, Dict, Union, Callable, Optional, Tuple
from functools import partial

# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ==========================================
# 0. Data Structures & Helpers
# ==========================================

class OutputGeometry:
    """
    Standardized output object for the pipeline.
    Compatible with downstream Layer 3 GeometryTrial.
    """
    def __init__(self, name, family, events, labels, weights, purity, auc, cluster_id=None, params=None):
        self.name = name
        self.family = family
        self.events = events
        self.labels = labels
        self.weights = weights
        self.purity = purity      # Uniqueness Score
        self.auc = auc            # Learnability Score (The Tournament Metric)
        self.cluster_id = cluster_id
        self.params = params or {}
    
    def __repr__(self):
        return f"<Geometry {self.name} | AUC={self.auc:.3f} | Purity={self.purity:.2f} | N={len(self.events)} | Cluster={self.cluster_id}>"

class KalmanFilter1D:
    """
    Simple 1D Kalman Filter for trend smoothing.
    Required by generate_dual_cusum_signals.
    """
    def __init__(self, Q: float = 1e-5, R: float = 0.01, initial_value: float = 0.0):
        self.Q = Q  # Process variance
        self.R = R  # Measurement variance
        self.x = initial_value  # State estimate
        self.P = 1.0  # Error covariance

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        values = series.values
        n = len(values)
        x_hat = np.zeros(n)
        P_hat = np.zeros(n)

        x = self.x
        P = self.P
        Q = self.Q
        R = self.R

        for i in range(n):
            # Prediction step
            x_pred = x
            P_pred = P + Q

            # Update step
            z = values[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred

            x_hat[i] = x
            P_hat[i] = P

        return pd.Series(x_hat, index=series.index), pd.Series(P_hat, index=series.index)

def roll_entropy(series: pd.Series, window: int = 24, bins: int = 10) -> pd.Series:
    def _entropy_calc(x):
        # Handle constant input
        if np.max(x) == np.min(x):
            return 0.0
        hist_counts, _ = np.histogram(x, bins=bins)
        return shannon_entropy(hist_counts)

    return series.rolling(window).apply(_entropy_calc, raw=True)

def calc_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    """
    pv = price * volume
    cum_pv = pv.rolling(window).sum()
    cum_vol = volume.rolling(window).sum()
    return cum_pv / (cum_vol + 1e-9)

def calc_tr(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    """
    Calculate True Range (TR).
    Uses High/Low if available in df, otherwise falls back to abs(diff(Close)).
    """
    # Case-insensitive check
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
        # Fallback
        tr = close.diff().abs()
    return tr

def average_uniqueness(indicator: pd.DataFrame) -> float:
    """Calculates the average uniqueness of a signal based on its indicator matrix."""
    concurrency = indicator.sum(axis=1)
    valid_c = concurrency[concurrency > 0]
    if valid_c.empty: return 0.0
    return (1.0 / valid_c).mean()

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    """
    Maps events to the full timeline as a binary indicator series.
    Marks the ENTIRE duration of the label (from t to t+horizon) as active.
    This ensures De Prado's uniqueness metric accounts for trade overlap duration.
    """
    # Create an empty integer array for speed
    arr = np.zeros(len(index), dtype=int)
    
    # Get integer locations of events
    # We intersection check first to ensure events are within index range
    valid_events = events.intersection(index)
    
    if valid_events.empty:
        return pd.DataFrame(0, index=index, columns=[0])

    # Convert timestamps to integer locations in the index
    # Note: searchsorted is fast but requires sorted index
    event_locs = index.get_indexer(valid_events)
    event_locs = event_locs[event_locs != -1] # Safety check
    
    # Mark durations
    # A simple loop is fast enough for ~500 events
    # For very large arrays, we could use difference array accumulation
    n_bars = len(index)
    for loc in event_locs:
        end_loc = min(loc + horizon, n_bars)
        arr[loc:end_loc] += 1
        
    # Any value > 0 means the strategy is "in the market"
    # We clamp to 1 because we are building a binary indicator of "Active Status"
    # The sum across geometries (concurrency) is calculated later
    arr = np.clip(arr, 0, 1)
    
    return pd.DataFrame(arr, index=index, columns=[0])

def normalized_mi(y1: pd.Series, y2: pd.Series) -> float:
    """
    Calculates Symmetric Normalized Mutual Information (0 to 1).
    Uses min(H(X), H(Y)) as denominator to prevent bias against low-entropy signals.
    """
    common = y1.index.intersection(y2.index)
    if len(common) < 30: 
        return 0.0

    mi = mutual_info_score(y1.loc[common], y2.loc[common])
    
    h1 = shannon_entropy(y1.loc[common].value_counts())
    h2 = shannon_entropy(y2.loc[common].value_counts())
    
    denom = min(h1, h2)
    return mi / denom if denom > 0 else 0.0

def check_class_balance(labels: pd.Series, min_class_samples: int = 20) -> Dict:
    counts = labels.value_counts()
    total = len(labels)
    n_buy = counts.get(1, 0)
    n_sell = counts.get(-1, 0)
    n_signals = n_buy + n_sell

    valid = True
    if n_signals < min_class_samples:
        valid = False

    return {
        "valid": valid,
        "total": total,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "balance_ratio": n_signals / total if total > 0 else 0
    }

# ==========================================
# 1. Event Generators (New Orthogonal Families)
# ==========================================

class BaseEventGenerator:
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class EntropyEvents(BaseEventGenerator):
    """
    NEW FAMILY: Structural.
    Triggers when the 'information content' (Entropy) of the price action spikes.
    This is highly orthogonal to standard volatility or trend signals.
    """
    def generate(self, price: pd.Series, window: int = 24, z_thresh: float = 2.0) -> pd.DatetimeIndex:
        log_ret = np.log(price).diff().fillna(0)
        # Calculate rolling entropy of returns
        ent = roll_entropy(log_ret, window=window, bins=10)

        # Z-Score of entropy
        ent_mean = ent.rolling(window*5).mean()
        ent_std = ent.rolling(window*5).std()
        z_ent = (ent - ent_mean) / (ent_std + 1e-6)

        # Trigger on structural break (high entropy change)
        trigger = (z_ent > z_thresh) & (z_ent.shift(1) <= z_thresh)
        return price.index[trigger]


class MicrostructureEvents(BaseEventGenerator):
    """
    NEW FAMILY: Liquidity/Microstructure.
    Uses Amihud Illiquidity proxy (AbsRet / Volume) to find liquidity gaps.
    """
    def generate(self, df: pd.DataFrame, window: int = 20, z: float = 2.0) -> pd.DatetimeIndex:
        # Requires Volume
        if 'volume' not in df.columns: return pd.DatetimeIndex([])

        ret = df['close'].pct_change().abs()
        amihud = ret / (df['volume'] * df['close'] + 1e-9)

        # Rolling Z-Score of Illiquidity
        mu = amihud.rolling(window).mean()
        sigma = amihud.rolling(window).std()
        z_score = (amihud - mu) / (sigma + 1e-9)

        trigger = z_score > z
        return df.index[trigger]


class TrendModulatedBreakoutEvents(BaseEventGenerator):
    """
    Detects Donchian Channel Breakouts with Trend Modulation.
    Only allows Longs if price > Anchor, Shorts if price < Anchor.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 20, anchor_window: int = 100) -> pd.DatetimeIndex:
        price = df['close']

        # Donchian Channel Breakout
        rolling_max = price.rolling(lookback).max().shift(1)
        rolling_min = price.rolling(lookback).min().shift(1)

        # Anchor (Trend Filter)
        # Use VWAP if volume available, else SMA
        if 'volume' in df.columns:
            anchor = calc_vwap(price, df['volume'], anchor_window)
        else:
            anchor = price.rolling(anchor_window).mean()

        breakout_high = (price > rolling_max) & (price > anchor)
        breakout_low = (price < rolling_min) & (price < anchor)

        breakout = breakout_high | breakout_low
        # Initiation only
        event = breakout & ~breakout.shift(1).fillna(False)
        return price.index[event]


class KalmanTrendEvents(BaseEventGenerator):
    """
    Replaces SMA Crossovers with Kalman Filter Crossovers (Fast vs Slow).
    Significantly reduces lag.
    """
    def generate(self, price: pd.Series, q_fast: float = 1e-3, q_slow: float = 1e-5) -> pd.DatetimeIndex:
        # Fast Filter
        kf_fast = KalmanFilter1D(Q=q_fast, R=0.01, initial_value=price.iloc[0])
        fast_line, _ = kf_fast.filter_series(price)

        # Slow Filter
        kf_slow = KalmanFilter1D(Q=q_slow, R=0.01, initial_value=price.iloc[0])
        slow_line, _ = kf_slow.filter_series(price)

        # Crossover Logic
        cross = (fast_line > slow_line) & (fast_line.shift(1) <= slow_line.shift(1))
        # Also could detect bearish cross: (fast_line < slow_line) & (fast_line.shift(1) >= slow_line.shift(1))
        # But usually TrendInitiation implies checking both or specific direction.

        cross_bull = (fast_line > slow_line) & (fast_line.shift(1) <= slow_line.shift(1))
        cross_bear = (fast_line < slow_line) & (fast_line.shift(1) >= slow_line.shift(1))

        return price.index[cross_bull | cross_bear]


class ATRShockEvents(BaseEventGenerator):
    """
    Volatility Shock using ATR Normalization instead of StdDev.
    Captures Gap volatility.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 14, long_window: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        price = df['close']
        tr = calc_tr(df, price)

        atr = tr.rolling(lookback).mean()
        atr_mean = atr.rolling(long_window).mean()
        atr_std = atr.rolling(long_window).std()

        z_score = (atr - atr_mean) / (atr_std + 1e-9)

        trigger = z_score > z
        return price.index[trigger]


class VWAPReversionEvents(BaseEventGenerator):
    """
    Mean Reversion using VWAP as the anchor.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        price = df['close']
        if 'volume' not in df.columns: return pd.DatetimeIndex([])

        vwap = calc_vwap(price, df['volume'], lookback)
        std = price.rolling(lookback).std()

        zscore = (price - vwap) / (std + 1e-6)
        return price.index[np.abs(zscore) > z]


class KalmanRegimeEvents(BaseEventGenerator):
    """
    Triggers when price deviates significantly from the Kalman trend estimate.
    Parameters modeled after label_based_layer_0 optimization ranges.
    """
    def generate(self, price: pd.Series, Q: float = 1e-4, R: float = 0.01, z: float = 2.0) -> pd.DatetimeIndex:
        kf = KalmanFilter1D(Q=Q, R=R, initial_value=price.iloc[0])
        trend, _ = kf.filter_series(price)

        # Deviation
        diff = price - trend
        # We need a dynamic threshold for deviation.
        # Using rolling std of the diff itself

        std = diff.rolling(20).std()
        zscore = diff / (std + 1e-9)

        # Trigger on extreme deviation
        return price.index[np.abs(zscore) > z]


class VWAPCrossEvents(BaseEventGenerator):
    """
    Triggers when price crosses the VWAP (Liquidity/Value validation).
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50) -> pd.DatetimeIndex:
        price = df['close']
        if 'volume' not in df.columns: return pd.DatetimeIndex([])

        vwap = calc_vwap(price, df['volume'], lookback)

        cross_up = (price > vwap) & (price.shift(1) <= vwap.shift(1))
        cross_down = (price < vwap) & (price.shift(1) >= vwap.shift(1))

        return price.index[cross_up | cross_down]


class AdaptiveSymmetricCUSUMEvents(BaseEventGenerator):
    """
    Symmetric CUSUM with Dynamic Thresholds based on Volatility.
    """
    def generate(self, price: pd.Series, multiplier: float = 0.5, vol_window: int = 20) -> pd.DatetimeIndex:
        t_events = []
        s_pos = 0
        s_neg = 0

        diff = np.log(price).diff()
        vol = diff.rolling(vol_window).std()

        # Vectorized loop preparation
        diff_val = diff.values
        vol_val = vol.values
        idx = price.index

        # Handle nans at start
        start_idx = vol_window
        if np.isnan(vol_val[start_idx]):
            valid_indices = np.where(~np.isnan(vol_val))[0]
            if len(valid_indices) > 0:
                start_idx = valid_indices[0]
            else:
                return pd.DatetimeIndex([])

        for i in range(start_idx, len(price)):
            h = vol_val[i] * multiplier # Dynamic Threshold
            if np.isnan(h) or h == 0: continue

            r = diff_val[i]
            if np.isnan(r): continue

            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            if s_pos > h:
                s_neg = 0; s_pos = 0
                t_events.append(idx[i])
            elif s_neg < -h:
                s_neg = 0; s_pos = 0
                t_events.append(idx[i])

        return pd.DatetimeIndex(t_events)


# ==========================================
# 2. Labeling Logic (Trailing Stop Vectorized)
# ==========================================

def get_price_path_matrix(df: pd.DataFrame, events: pd.DatetimeIndex, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts (N_events, Horizon) matrices for High, Low, and Close prices using fancy indexing.
    Starts from t+1 (next bar) to avoid lookahead bias of the signal bar itself.
    """
    n_events = len(events)
    # Get integer locations
    event_idx = df.index.get_indexer(events)

    # Filter out events near the end
    valid_mask = (event_idx != -1) & (event_idx < len(df) - horizon - 1)
    event_idx = event_idx[valid_mask]

    if len(event_idx) == 0:
        return np.array([]), np.array([]), np.array([])

    # Create indices matrix: (N, Horizon)
    # Start checking from t+1
    idx_matrix = event_idx[:, None] + np.arange(1, horizon + 1)

    # Extract
    if 'high' in df.columns and 'low' in df.columns:
        highs = df['high'].values[idx_matrix]
        lows = df['low'].values[idx_matrix]
    else:
        # Fallback to close if OHLC not available (though strongly discouraged)
        highs = df['close'].values[idx_matrix]
        lows = df['close'].values[idx_matrix]
        
    closes = df['close'].values[idx_matrix]

    return highs, lows, closes

def vectorized_trailing_stop_label(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    horizon: int = 120,
    gap: float = 2.0,
    sl_mult: float = 1.0,
    min_profit: float = 0.002
) -> pd.DataFrame:
    """
    Vectorized Trailing Stop Labeling.
    Implements a "Pessimistic" ratchet:
      1. Check Low against previous Stop.
      2. Update HWM with High.
      3. Update Stop based on HWM.
    """
    # 1. Prepare Data
    # Align volatility
    vol_events = volatility.reindex(events).fillna(method='bfill').fillna(0.01).values

    # Entry Prices (Close at signal time)
    entry_prices = df['close'].reindex(events).values

    # Filter valid events (handled in get_price_path but we need consistency)
    event_locs = df.index.get_indexer(events)
    valid_mask = (event_locs != -1) & (event_locs < len(df) - horizon - 1)

    if np.sum(valid_mask) == 0:
        return pd.DataFrame()

    events = events[valid_mask]
    vol_events = vol_events[valid_mask]
    entry_prices = entry_prices[valid_mask]

    # Get Future Paths (N, H)
    highs, lows, closes = get_price_path_matrix(df, events, horizon)

    if highs.size == 0:
        return pd.DataFrame()

    n_events, h = highs.shape

    # 2. Compute Initial Stops
    # Initial Stop = Entry * (1 - SL * Vol)
    initial_stops = entry_prices * (1 - sl_mult * vol_events)

    # 3. Compute High Water Marks (HWM)
    # We include Entry Price in HWM calculation to ensure HWM starts at least at Entry
    # Concatenate Entry to Highs to compute cumulative max, then slice off Entry
    # Shape: (N, H+1)
    augmented_highs = np.column_stack((entry_prices, highs))
    hwm_stream = np.maximum.accumulate(augmented_highs, axis=1)

    # 4. Compute Ratchet Stops
    # Potential Stop = HWM * (1 - Gap * Vol)
    # We broadcast vol_events to (N, 1)
    gap_pct = (vol_events * gap)[:, None]
    potential_stops = hwm_stream * (1 - gap_pct)

    # Ratchet: Stop can only go UP.
    ratchet_stops_stream = np.maximum.accumulate(potential_stops, axis=1)

    # Floor with Initial Stop
    # Broadcast Initial Stops to (N, H+1)
    ratchet_stops_stream = np.maximum(ratchet_stops_stream, initial_stops[:, None])

    # 5. Check Exits (Pessimistic)
    # At step k (bar t+1+k), we check Low[k] against Stop from previous step.
    # Stop from previous step is ratchet_stops_stream[:, k] (index k corresponds to state after bar k-1)
    # Note: ratchet_stops_stream column 0 is derived from Entry Price.
    # This is the stop active for the first bar (Highs/Lows column 0).
    # So we use columns 0 to H-1 of Ratchet Stream as effective stops for bars 0 to H-1.
    effective_stops = ratchet_stops_stream[:, :-1]

    # Check Hits: Low <= Effective Stop
    hits = lows <= effective_stops

    # Find first hit index
    # argmax returns index of first True. If no True, returns 0.
    first_hit_idx = np.argmax(hits, axis=1)

    # Check if there was actually a hit (if argmax=0, check if index 0 is True)
    # Or simpler: verify any hit in the row
    has_hit = np.any(hits, axis=1)

    # 6. Determine Exit Price and PnL
    exit_prices = np.zeros(n_events)
    # exit_indices = np.zeros(n_events, dtype=int)

    # For hits: Exit Price is the Stop Level triggered
    # We take effective_stops[row, hit_idx]
    # Use fancy indexing
    row_indices = np.arange(n_events)

    # Default (No Hit): Exit at last Close (Time Expiry)
    exit_prices[~has_hit] = closes[~has_hit, -1]
    # exit_indices[~has_hit] = h - 1

    # For Hits:
    hit_rows = row_indices[has_hit]
    hit_cols = first_hit_idx[has_hit]
    exit_prices[has_hit] = effective_stops[hit_rows, hit_cols]
    # exit_indices[has_hit] = hit_cols

    # Calculate Returns
    returns = (exit_prices - entry_prices) / entry_prices

    # 7. Generate Labels and Weights
    # Label 1 if Return > min_profit, else -1 (or 0)
    labels = np.where(returns > min_profit, 1.0, -1.0)

    # Weighting (similar to TBM)
    # Weight by return magnitude relative to target?
    # Here target is infinite.
    # We can scale by volatility or Sharpe proxy.
    # Let's use Return / (Gap * Vol) as a proxy for "R-multiple" captured
    # (Gap*Vol is roughly the risk)
    risk_unit = gap_pct[:, 0] * entry_prices # Approx initial risk distance
    r_multiple = np.abs(returns * entry_prices) / (risk_unit + 1e-9)
    weights = np.clip(r_multiple, 0.1, 2.0)

    # Construct DataFrame
    out_df = pd.DataFrame({
        'label': labels,
        'ret': returns,
        'weight': weights
    }, index=events)

    return out_df

# ==========================================
# 3. Probe & Validation Tools (Purged CV)
# ==========================================

class RobustFocalLoss:
    """
    Production-grade Focal Loss for LightGBM in Financial ML.
    """

    def __init__(
        self,
        gamma_pos=1.0, # gamma_fn: Preference for Opportunity (Missed Upside)
        gamma_neg=2.5, # gamma_fp: Preference for Safety (Traps)
        alpha=None,
        grad_clip=5.0,
        w_cap=3.0,
        mix=0.25,
        label_smoothing=0.02,
        verbose=True
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.grad_clip = grad_clip
        self.w_cap = w_cap
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.verbose = verbose
        self._is_init = False

    def _init_alpha(self, labels):
        """Auto-compute alpha based on prevalence if not provided."""
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                # Standard inverse frequency: High alpha for rare positives
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5

        # Clamp alpha for safety
        self.alpha = np.clip(self.alpha, 0.05, 0.95)

        if self.verbose:
            logger.info(f"[LGBM Focal] Gamma(+):{self.gamma_pos} Gamma(-):{self.gamma_neg} | Alpha:{self.alpha:.4f}")

        self._is_init = True

    def __call__(self, preds, train_data):
        if hasattr(train_data, 'get_label'):
             labels = train_data.get_label()
        else:
             labels = train_data

        # Lazy init alpha on first call to handle data loading
        if not self._is_init:
            self._init_alpha(labels)

        # 1. Label Smoothing (Crucial for Finance)
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2. Robust Sigmoid
        p = expit(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # 3. Vectorized Asymmetric Gamma
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)

        # 4. Focal Weights with Capping
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        # 5. Gradient & Hessian Calculation
        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce
        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        # 6. Mixing (Stability Anchor)
        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        # 7. Clipping & Safety
        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6) # Prevent divide-by-zero

        return grad, hess

def generate_probe_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Generates a standardized 'Basis Set' of features for Geometry Validation.
    These use fixed industry-standard lookbacks to serve as a robust benchmark.
    """
    df = pd.DataFrame(index=price.index)

    # 1. Momentum (Immediate & Short-term)
    df['ret_1'] = np.log(price).diff(1)
    df['ret_12'] = np.log(price).diff(12) # Context momentum

    # 2. Oscillator (RSI 14)
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 3. Volatility Regime (20 vs 100)
    # Is recent vol expanding relative to history?
    vol_20 = df['ret_1'].rolling(20).std()
    vol_100 = df['ret_1'].rolling(100).std()
    df['vol_ratio'] = vol_20 / (vol_100 + 1e-6)

    # 4. Trend Distance (50 bar MA)
    # Are we far from the mean?
    ma_50 = price.rolling(50).mean()
    df['trend_dist'] = (price / ma_50) - 1

    # 5. Liquidity Shock (Volume vs 20 bar avg)
    if volume is not None:
        vol_ma_20 = volume.rolling(20).mean()
        df['vol_shock'] = volume / (vol_ma_20 + 1e-6)

    # Clean up (Probe models hate NaNs)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def get_purged_cv_score(X, y, w, horizon_bars=48) -> float:
    """
    Purged CV Score (AUC) calculation.
    """
    if len(y) < 50: return 0.5
    
    # Check for empty features
    if X.shape[1] == 0:
        return 0.5

    n_splits = 3
    # Gap must encompass the entire label horizon to prevent leakage
    gap = int(1.1 * horizon_bars) + 2 
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    
    focal_loss = RobustFocalLoss(gamma_pos=1.0, gamma_neg=2.0, verbose=False)

    params = {
        'objective': focal_loss,
        'metric': 'auc',
        'verbosity': -1,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'is_unbalance': True
    }
    
    # Remap labels to binary
    y_binary = (y > 0).astype(int)

    if len(y) < 100: 
        split_idx = int(len(y) * 0.7)
        tr_idx, va_idx = np.arange(split_idx), np.arange(split_idx + gap, len(y))
        splits = [(tr_idx, va_idx)]
    else:
        splits = tscv.split(X)
    
    valid_splits_count = 0
    
    for tr_idx, va_idx in splits:
        if len(tr_idx) < 20 or len(va_idx) < 20: continue
        
        curr_X_tr, curr_X_tr_valid = X.iloc[tr_idx], X.iloc[va_idx]
        curr_y_tr, curr_y_tr_valid = y_binary.iloc[tr_idx], y_binary.iloc[va_idx]
        curr_w_tr, curr_w_tr_valid = w.iloc[tr_idx], w.iloc[va_idx]
        
        # Check if valid set has both classes
        if len(curr_y_tr_valid.unique()) < 2:
             continue

        dtrain = lgb.Dataset(curr_X_tr, label=curr_y_tr, weight=curr_w_tr)
        dvalid = lgb.Dataset(curr_X_tr_valid, label=curr_y_tr_valid, weight=curr_w_tr_valid)
        
        try:
            model = lgb.train(params, dtrain, valid_sets=[dvalid],
                              callbacks=[lgb.early_stopping(10, verbose=False)])

            score = model.best_score['valid_0']['auc']
            scores.append(score)
            valid_splits_count += 1
        except Exception as e:
            pass
            
    return np.mean(scores) if valid_splits_count > 0 else 0.5

# ==========================================
# 4. Clustering & Selection
# ==========================================

def cluster_geometries(geometries: List[Dict]) -> Dict[str, int]:
    """
    Uses Hierarchical Clustering (ONC-style) to group geometries.
    Returns a dict mapping {name: cluster_id}.
    """
    if not geometries:
        return {}

    # 1. Create correlation matrix of labels
    # We need a common index
    all_dates = sorted(list(set().union(*[g['labels'].index for g in geometries])))
    
    df_labels = pd.DataFrame(index=all_dates)
    for g in geometries:
        # Realign to common index, fill NA with 0 (neutral)
        df_labels[g['name']] = g['labels'].reindex(all_dates).fillna(0)

    # Correlation matrix (Spearman for non-linear dependency)
    corr_mat = df_labels.corr(method='spearman').fillna(0)
    
    # Distance matrix
    dist_mat = np.sqrt(0.5 * (1 - corr_mat)).clip(0, 1) # Metric distance
    dist_mat = dist_mat.fillna(1.0) # Safety
    
    # Hierarchical Clustering
    try:
        # squareform is needed if input is a distance matrix
        linkage = sch.linkage(squareform(dist_mat.values, checks=False), method='ward')
        
        # Max distance threshold for clusters (e.g., dist=0.7 implies corr ~0)
        # Higher threshold = fewer clusters
        cluster_ids = sch.fcluster(linkage, t=0.7, criterion='distance')
        
        return dict(zip(df_labels.columns, cluster_ids))
    except Exception as e:
        logger.warning(f"Clustering failed: {e}. Assigning all to cluster 1.")
        return {g['name']: 1 for g in geometries}

# ==========================================
# 5. Main Pipeline
# ==========================================

def orthogonal_label_generation(
    price: pd.Series,
    volume: pd.Series,
    df_full: Optional[pd.DataFrame] = None,
    tau_auc: float = 0.52,
    tau_mi: float = 0.15,
    tau_uniq: float = 0.10
) -> List[OutputGeometry]:
    """
    Main Execution Pipeline for Orthogonal Label Generation.
    Replaces standard greedy selection with ONC-style clustering.
    Implements Trailing Stop Logic with Grid Search.
    """
    logger.info("--- Starting Advanced Geometry Generation ---")
    
    # 1. Generate Probe Features
    X_probe = generate_probe_features(price, volume)
    
    # Use df_full if provided, else construct min necessary
    if df_full is None:
        # Warning: Using Close for High/Low if df_full is not provided
        logger.warning("df_full not provided. Using Close for High/Low path simulation.")
        df_full = pd.DataFrame({'close': price, 'high': price, 'low': price, 'volume': volume})
    elif 'volume' not in df_full.columns and volume is not None:
        df_full['volume'] = volume

    # 2. Define Candidates (Generators)
    # Including multiple variations to feed the clustering
    generators = [
        # Structural Family (Entropy)
        ('ENTROPY', EntropyEvents(), {'window': 24}),
        ('ENTROPY_FAST', EntropyEvents(), {'window': 12}),
        ('ENTROPY_SLOW', EntropyEvents(), {'window': 48}),
        
        # Volatility Family (Adaptive CUSUM Upgrade)
        ('CUSUM_ADAPT', AdaptiveSymmetricCUSUMEvents(), {'multiplier': 0.5, 'vol_window': 50}),
        ('CUSUM_TIGHT', AdaptiveSymmetricCUSUMEvents(), {'multiplier': 0.3, 'vol_window': 20}),
        ('CUSUM_LOOSE', AdaptiveSymmetricCUSUMEvents(), {'multiplier': 0.8, 'vol_window': 50}),
        
        # Liquidity Family
        ('LIQUIDITY', MicrostructureEvents(), {'window': 20}),
        ('LIQUIDITY_FAST', MicrostructureEvents(), {'window': 10}),
        
        # Breakout Family (Trend Modulated)
        ('BREAKOUT', TrendModulatedBreakoutEvents(), {'lookback': 48, 'anchor_window': 100}),
        ('BREAKOUT_FAST', TrendModulatedBreakoutEvents(), {'lookback': 20, 'anchor_window': 50}),

        # Kalman Trend (Replaces MA Trend)
        ('KALMAN_TREND', KalmanTrendEvents(), {'q_fast': 1e-3, 'q_slow': 1e-5}),
        ('KALMAN_TREND_SENS', KalmanTrendEvents(), {'q_fast': 5e-3, 'q_slow': 5e-5}),

        # Volatility Shock (ATR Upgrade)
        ('VOL_SHOCK', ATRShockEvents(), {'lookback': 14, 'long_window': 50, 'z': 2.0}),
        ('VOL_SHOCK_EXT', ATRShockEvents(), {'lookback': 14, 'long_window': 100, 'z': 3.0}),

        # Mean Reversion (VWAP Upgrade)
        ('MR_VWAP', VWAPReversionEvents(), {'lookback': 50, 'z': 2.5}),
        ('MR_VWAP_FAST', VWAPReversionEvents(), {'lookback': 24, 'z': 2.0}),

        # Kalman Regime (New)
        ('KALMAN_REGIME', KalmanRegimeEvents(), {'Q': 1e-4, 'R': 0.01, 'z': 2.0}),

        # VWAP Cross (New)
        ('VWAP_CROSS', VWAPCrossEvents(), {'lookback': 50}),
    ]
    
    # 3. Labeling Grid (Trailing Gap x Stop Loss)
    # Grid: Gap [1.5, 2.0, 2.5, 3.0] x SL [0.4, 0.6]
    trailing_gaps = [1.5, 2.0, 2.5, 3.0]
    stop_loss_mults = [0.4, 0.6]
    labeling_grid = list(product(trailing_gaps, stop_loss_mults))

    # 4. Process Candidates
    candidates = []
    volatility = price.pct_change().rolling(20).std()
        
    for name, gen, params in generators:
        # A. Generate Events
        try:
            if isinstance(gen, (MicrostructureEvents, TrendModulatedBreakoutEvents,
                                ATRShockEvents, VWAPReversionEvents, VWAPCrossEvents)):
                events = gen.generate(df_full, **params)
            else:
                events = gen.generate(price, **params)
        except Exception as e:
            logger.warning(f"Generator {name} failed: {e}")
            continue
            
        if len(events) < 30: continue

        # B. Loop over Labeling Grid (Trailing Stop)
        for gap, sl in labeling_grid:
            grid_name = f"{name}_G{gap}_S{sl}"

            # Trailing Stop Labeling (Vectorized)
            h = 120 # Standard horizon for Trailing Stop to play out
            labeled_df = vectorized_trailing_stop_label(
                df_full, events, volatility, horizon=h, gap=gap, sl_mult=sl
            )

            if labeled_df.empty: continue

            # Check Class Balance
            balance_check = check_class_balance(labeled_df['label'])
            if not balance_check['valid']:
                 continue

            # C. Probe (Learnability Check)
            y = labeled_df['label']
            w = labeled_df['weight']

            # Align probe features to events
            X_curr = X_probe.loc[y.index]

            try:
                auc = get_purged_cv_score(X_curr, y, w, horizon_bars=h)
            except Exception:
                auc = 0.5

            candidates.append({
                'name': grid_name,
                'family': name.split('_')[0],
                'events': events,
                'labels': y,
                'weights': w,
                'auc': auc,
                'indicator': build_indicator_matrix(events, price.index, horizon=h),
                'params': {**params, 'gap': gap, 'sl_mult': sl, 'horizon': h}
            })

        logger.info(f"Processed {name}, Total Candidates: {len(candidates)}")

    if not candidates:
        return []

    # 5. Cluster for Orthogonality (The "Teacher" Selection)
    logger.info("--- Clustering Geometries ---")
    cluster_map = cluster_geometries(candidates)
    
    # Group by cluster
    clusters = {}
    for cand in candidates:
        c_id = cluster_map[cand['name']]
        if c_id not in clusters: clusters[c_id] = []
        clusters[c_id].append(cand)
        cand['cluster_id'] = c_id

    # 6. Selection (Best AUC per Cluster)
    final_geometries = []

    for c_id, group in clusters.items():
        # Pick the best AUC in this cluster
        best_in_cluster = max(group, key=lambda x: x['auc'])

        # Purity Calculation
        purity = average_uniqueness(best_in_cluster['indicator'])

        geo = OutputGeometry(
            best_in_cluster['name'],
            best_in_cluster['family'],
            best_in_cluster['events'],
            best_in_cluster['labels'],
            best_in_cluster['weights'],
            purity,
            best_in_cluster['auc'],
            cluster_id=c_id,
            params=best_in_cluster['params']
        )

        # Threshold
        if geo.auc > tau_auc:
            final_geometries.append(geo)

    logger.info(f"Selected {len(final_geometries)} orthogonal geometries from {len(clusters)} clusters.")
    return final_geometries
