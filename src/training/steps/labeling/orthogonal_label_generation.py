import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import entropy as shannon_entropy
from scipy.special import expit
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
    def __init__(self, name, family, events, labels, weights, purity, auc):
        self.name = name
        self.family = family
        self.events = events
        self.labels = labels
        self.weights = weights
        self.purity = purity      # Uniqueness Score
        self.auc = auc            # Learnability Score (The Tournament Metric)
    
    def __repr__(self):
        return f"<Geometry {self.name} | AUC={self.auc:.3f} | Purity={self.purity:.2f} | N={len(self.events)}>"

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

def average_uniqueness(indicators: pd.DataFrame) -> float:
    """
    Calculates average uniqueness (1 / concurrency) across all events.
    Matches AFML Ch. 4 logic exactly.
    """
    if indicators.empty:
        return 0.0

    concurrency = indicators.sum(axis=1)
    # Avoid div by zero
    uniqueness = indicators.div(concurrency, axis=0).fillna(0)

    # only count rows where this geometry is active
    mask = indicators > 0
    uniq_vals = uniqueness[mask]

    if uniq_vals.count().sum() == 0:
        return 0.0

    return uniq_vals.mean().mean()

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

def label_distribution_stable(labels: pd.Series, splits: int = 5, eps: float = 0.15) -> bool:
    """
    Checks if label distribution is stationary across time chunks.
    """
    if len(labels) < splits * 10: 
        return True 

    labels = labels.sort_index()
    chunks = np.array_split(labels, splits)
    
    for a, b in combinations(chunks, 2):
        if len(a) < 10 or len(b) < 10:
            continue
            
        pa = a.value_counts(normalize=True)
        pb = b.value_counts(normalize=True)
        pa, pb = pa.align(pb, fill_value=0)
        
        d = shannon_entropy(pa, pb)
        if not np.isfinite(d): 
             d = 1.0
             
        if d > eps:
            return False
    return True

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
# 1. Event Generators (The 7 Families + Controls)
# ==========================================

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
    Generate dual CUSUM signals for trend-following and mean-reversion using optimized Kalman filter.
    """
    # 1. Compute log returns
    log_ret = np.log(close / close.shift(1)).fillna(0.0)

    # 2. Apply 1D Kalman filter
    kf = KalmanFilter1D(Q=Q, R=R, initial_value=float(log_ret.iloc[0]))
    log_ret_smooth_raw, _ = kf.filter_series(log_ret)

    # Ensure it's a series with correct index for rolling operations
    if not isinstance(log_ret_smooth_raw, pd.Series):
        log_ret_smooth_series = pd.Series(log_ret_smooth_raw, index=close.index).fillna(0.0)
    else:
        log_ret_smooth_series = log_ret_smooth_raw.fillna(0.0)

    # 3. Rolling volatility & ER (Vectorized)
    sigma = log_ret_smooth_series.rolling(window_vol, min_periods=1).std()

    # Efficiency Ratio calculation
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

    # 6. CUSUM Loop (Optimized Numpy)
    n = len(close)

    # Convert to numpy for speed
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

        # Trend CUSUM (on smoothed returns)
        S_trend_pos = max(0.0, S_trend_pos + r_arr[t])
        S_trend_neg = min(0.0, S_trend_neg + r_arr[t])

        if S_trend_pos > cur_h:
            trend_signal[t] = 1
            S_trend_pos = 0.0
        elif S_trend_neg < -cur_h:
            trend_signal[t] = -1
            S_trend_neg = 0.0

        # Reversal CUSUM (Mean Reversion on Residuals)
        S_rev_pos = max(0.0, S_rev_pos + res_arr[t])
        S_rev_neg = min(0.0, S_rev_neg + res_arr[t])

        if S_rev_pos > cur_h:
            reversal_signal[t] = 1 # Overextended UP -> Expect Reversal
            S_rev_pos = 0.0
        elif S_rev_neg < -cur_h:
            reversal_signal[t] = -1 # Overextended DOWN -> Expect Reversal
            S_rev_neg = 0.0

    # Pack results
    signals = pd.DataFrame({
        'trend_signal': trend_signal,
        'reversal_signal': reversal_signal,
        'h_t': h_t,
        'er': ER
    }, index=close.index)

    return signals

class BaseEventGenerator:
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

# --- CONTROL GROUPS (NULL HYPOTHESES) ---
class RandomEvents(BaseEventGenerator):
    """
    Null Hypothesis 1: Random Sampling.
    """
    def generate(self, price: pd.Series, n_events: int = 100) -> pd.DatetimeIndex:
        if len(price) < n_events: n_events = len(price)
        rng = np.random.default_rng(42) 
        random_indices = rng.choice(price.index, size=n_events, replace=False)
        return pd.DatetimeIndex(np.sort(random_indices))

class TimeEvents(BaseEventGenerator):
    """
    Null Hypothesis 2: Time-based sampling.
    """
    def generate(self, price: pd.Series, step: int = 50) -> pd.DatetimeIndex:
        return price.index[::step]

# --- ANTI-BIAS FAMILIES (REGIME BALANCE) ---
class LowVolatilityEvents(BaseEventGenerator):
    """
    Triggers when volatility is exceptionally LOW (Bottom Quantile).
    """
    def generate(self, price: pd.Series, lookback: int = 50, quantile: float = 0.20) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        thresh = vol.rolling(lookback * 5).quantile(quantile)
        trigger = (vol < thresh) & (vol.shift(1) >= thresh.shift(1))
        return price.index[trigger]

class ChopEvents(BaseEventGenerator):
    """
    Triggers in Trendless/Choppy markets. Efficiency Ratio (ER) < Threshold.
    """
    def generate(self, price: pd.Series, lookback: int = 20, er_thresh: float = 0.3) -> pd.DatetimeIndex:
        change = price.diff(lookback).abs()
        path = price.diff().abs().rolling(lookback).sum()
        er = change / (path + 1e-6)
        trigger = (er < er_thresh) & (er.shift(1) >= er_thresh)
        return price.index[trigger]

# --- STANDARD FAMILIES ---
class VolatilityShockEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.0, use_quantile: bool = False, q: float = 0.95) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        
        if use_quantile:
            thresh = vol.rolling(lookback*5).quantile(q)
            trigger = vol > thresh
            return price.index[trigger]
        else:
            vol_mean = vol.expanding(min_periods=lookback).mean()
            vol_std = vol.expanding(min_periods=lookback).std()
            zscore = (vol - vol_mean) / (vol_std + 1e-6)
            return price.index[zscore > z]

class TrendInitiationEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, short: int = 20, long: int = 100) -> pd.DatetimeIndex:
        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()
        cross = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
        return price.index[cross]

class BreakoutEvents(BaseEventGenerator):
    """
    Detects Donchian Channel Breakouts.
    Updated to support splitting Long (High) and Short (Low) breakouts for orthogonality.
    """
    def generate(self, price: pd.Series, lookback: int = 20, side: str = 'both') -> pd.DatetimeIndex:
        rolling_max = price.rolling(lookback).max().shift(1)
        rolling_min = price.rolling(lookback).min().shift(1)
        
        breakout_high = price > rolling_max
        breakout_low = price < rolling_min
        
        # Filter for initiation only
        event_high = breakout_high & ~breakout_high.shift(1).fillna(False)
        event_low = breakout_low & ~breakout_low.shift(1).fillna(False)
        
        if side == 'long':
            return price.index[event_high]
        elif side == 'short':
            return price.index[event_low]
        else:
            return price.index[event_high | event_low]

class MeanReversionExtremeEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        mean = price.rolling(lookback).mean()
        std = price.rolling(lookback).std()
        zscore = (price - mean) / (std + 1e-6)
        return price.index[np.abs(zscore) > z]

class LiquidityShockEvents(BaseEventGenerator):
    def generate(self, volume: pd.Series, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        vol_mean = volume.expanding(min_periods=lookback).mean()
        vol_std = volume.expanding(min_periods=lookback).std()
        zscore = (volume - vol_mean) / (vol_std + 1e-6)
        return volume.index[zscore > z]

class SymmetricCusumEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, h: float = 0.01) -> pd.DatetimeIndex:
        t_events = []
        s_pos = 0
        s_neg = 0
        diff = np.log(price).diff().dropna()
        for i in diff.index:
            r = diff.loc[i]
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)
            if s_pos > h:
                s_neg = 0; s_pos = 0
                t_events.append(i)
            elif s_neg < -h:
                s_neg = 0; s_pos = 0
                t_events.append(i)
        return pd.DatetimeIndex(t_events)

class ImprovedCUSUMEvents(BaseEventGenerator):
    """
    Wrapper for existing CUSUM filter logic from Layer 2.
    Implemented locally to avoid circular dependencies.
    """
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        # Default Params matching generate_primary_signals defaults
        k = params.get('k', 0.12)
        vol_window = params.get('vol_window', 20)
        er_window = params.get('er_window', 10)
        er_min = params.get('er_min', 0.2)
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        w_trend = params.get('w_trend', 1.0)
        w_reversal = params.get('w_reversal', 1.0)
        
        # Extract series
        close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        
        volume = None
        if 'volume' in df.columns:
            volume = df['volume']
        elif 'Volume' in df.columns:
            volume = df['Volume']
            
        try:
            dual_signals = generate_dual_cusum_signals(
                close=close,
                volume=volume,
                k=k,
                alpha=alpha,
                beta=beta,
                er_min=er_min,
                window_vol=vol_window,
                window_er=er_window,
                Q=1e-5,
                R=0.01
            )
            
            # Compute Composite Signal
            composite = (
                w_trend * dual_signals['trend_signal'] +   # Trend is primary direction
                w_reversal * dual_signals['reversal_signal'] # Reversal adds to it
            )
            
            return composite.index[composite != 0]
            
        except Exception as e:
            logger.warning(f"Improved CUSUM failed for params {params}: {e}. Skipping geometry.")
            return pd.DatetimeIndex([])


class HurstStateEvents(BaseEventGenerator):
    def _get_hurst_exponent(self, ts):
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def generate(self, price: pd.Series, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        # Step optimization for speed
        hurst = price.rolling(lookback, step=5).apply(self._get_hurst_exponent, raw=True)
        # Forward fill carefully to avoid looking ahead (ffill propagates past value forward)
        hurst = hurst.reindex(price.index).ffill() 
        trigger = (hurst > threshold) & (hurst.shift(1) <= threshold)
        return price.index[trigger]

# ==========================================
# 2. Labeling Logic (Dynamic & Vol-Aware)
# ==========================================

def dynamic_mae_mfe_label(price: pd.Series, events: pd.DatetimeIndex, 
                          volatility: pd.Series, 
                          horizon: int = 24, 
                          min_ret_factor: float = 0.5, 
                          min_ret_floor: float = 0.002, 
                          dominance_ratio: float = 1.5) -> pd.DataFrame:
    results = {}
    price_arr = price.values
    valid_vol = volatility.reindex(events).fillna(0.01)
    
    for i, t in enumerate(events):
        if t not in price.index: continue
        t_idx = price.index.get_loc(t)
        if t_idx + horizon >= len(price): continue
            
        path = price_arr[t_idx : t_idx + horizon + 1]
        entry = path[0]
        returns = (path / entry) - 1.0
        mfe = np.max(returns)
        mae = np.min(returns) # negative
        
        long_ratio = mfe / (abs(mae) + 1e-6)
        short_ratio = abs(mae) / (mfe + 1e-6)
        
        lbl = 0
        weight = 1.0
        
        current_vol = valid_vol.iloc[i] if i < len(valid_vol) else 0.01
        dynamic_threshold = max(min_ret_floor, current_vol * min_ret_factor)

        if mfe > dynamic_threshold and long_ratio >= dominance_ratio:
            lbl = 1
            weight = np.log(1.0 + long_ratio)
        elif abs(mae) > dynamic_threshold and short_ratio >= dominance_ratio:
            lbl = -1
            weight = np.log(1.0 + short_ratio)
            
        if lbl != 0:
            results[t] = {'label': lbl, 'weight': weight}
            
    if not results: return pd.DataFrame()
    return pd.DataFrame.from_dict(results, orient='index')

def vol_scaled_fixed_label(price: pd.Series, events: pd.DatetimeIndex, 
                           horizon: int = 24, 
                           vol_lookback: int = 20,
                           z_threshold: float = 1.0) -> pd.DataFrame:
    results = {}
    returns = price.pct_change()
    vol = returns.rolling(vol_lookback).std()
    
    for t in events:
        if t not in price.index: continue
        t_idx = price.index.get_loc(t)
        if t_idx + horizon >= len(price): continue
        
        v_entry = vol.iloc[t_idx]
        if pd.isna(v_entry) or v_entry == 0: continue
            
        ret_horizon = (price.iloc[t_idx + horizon] / price.iloc[t_idx]) - 1.0
        threshold = v_entry * np.sqrt(horizon) * z_threshold
        
        lbl = 0
        weight = 1.0
        
        if ret_horizon > threshold:
            lbl = 1
            weight = abs(ret_horizon) / threshold
        elif ret_horizon < -threshold:
            lbl = -1
            weight = abs(ret_horizon) / threshold
            
        if lbl != 0:
            results[t] = {'label': lbl, 'weight': np.log(1.0 + weight)}
            
    if not results: return pd.DataFrame()
    return pd.DataFrame.from_dict(results, orient='index')

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
    # Simple pandas implementation or use TA-Lib
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
    vol_ma_20 = volume.rolling(20).mean()
    df['vol_shock'] = volume / (vol_ma_20 + 1e-6)

    # Clean up (Probe models hate NaNs)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def get_purged_lgbm_auc(X, y, w, horizon_bars=48) -> float:
    if len(y) < 50: return 0.5
    
    n_splits = 3
    # Gap must encompass the entire label horizon to prevent leakage
    gap = int(1.1 * horizon_bars) + 2 
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    
    # Updated: Use RobustFocalLoss
    focal_loss = RobustFocalLoss(gamma_pos=1.0, gamma_neg=2.0, verbose=False)

    params = {
        'objective': focal_loss,  # Custom objective
        'metric': 'auc',          # Metric compatible with binary classification
        'verbosity': -1,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'is_unbalance': True      # Handle imbalance explicitly
    }
    
    # Remap labels to binary (Predicting if we have a winning trade)
    y_binary = (y.abs() > 0).astype(int)

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
# 4. Selection Logic
# ==========================================

def select_best_geometries(candidates: List[Dict], tau_auc=0.55, tau_mi=0.15, tau_uniq=0.10) -> List[OutputGeometry]:
    # 1. SORT
    candidates.sort(
        key=lambda x: (
            -x['auc'],          
            -len(x['labels'])   
        )
    )

    # 1.1 Check MI between CUSUM variants if present
    cusum_variants = [c for c in candidates if "CUSUM" in c['family']]
    if len(cusum_variants) > 1:
        logger.info("\n--- CUSUM Variants MI Analysis ---")
        # Group by family+type to pick best repr for each variant
        # Actually just pairwise check all high AUC ones?
        # Let's check top 1 of each CUSUM subtype
        unique_cusum_types = {}
        for c in cusum_variants:
            t_key = f"{c['family']}_{c.get('name', '').split('_')[1]}" # e.g. CUSUM_IMP_TREND
            if t_key not in unique_cusum_types and c['auc'] > 0.52: # Only check reasonably good ones
                unique_cusum_types[t_key] = c

        keys = list(unique_cusum_types.keys())
        if len(keys) > 1:
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    mi = normalized_mi(unique_cusum_types[k1]['labels'], unique_cusum_types[k2]['labels'])
                    logger.info(f"MI({k1}, {k2}) = {mi:.4f}")
        logger.info("----------------------------------\n")
    
    accepted_configs = []
    accepted_objects = []
    global_indicator = pd.DataFrame() 
    
    logger.info(f"--- Starting Selection on {len(candidates)} Candidates ---")
    
    for cand in candidates:
        name = cand['name']
        
        # --- NULL HYPOTHESIS CHECK ---
        if cand['family'] == 'CONTROL':
            if cand['auc'] > 0.54: 
                logger.warning(f"⚠️  WARNING: Control Geometry {name} has High AUC ({cand['auc']:.3f}). Possible Leakage!")
            # Mark status for reporting
            cand['status'] = 'CONTROL'
            continue
            
        # A. Junk Filter
        if cand['auc'] < tau_auc:
            cand['status'] = 'REJECT_LOW_AUC'
            continue

        # B. Stability Filter
        if not label_distribution_stable(cand['labels']):
            logger.debug(f"Discard {name}: Unstable Labels")
            cand['status'] = 'REJECT_UNSTABLE'
            continue
            
        # C. Redundancy Filter
        is_redundant = False
        for acc in accepted_configs:
            mi_score = normalized_mi(cand['labels'], acc['labels'])
            if mi_score > tau_mi:
                logger.debug(f"Discard {name}: Redundant with {acc['name']} (MI={mi_score:.2f})")
                is_redundant = True
                cand['status'] = f'REJECT_REDUNDANT_{acc["name"]}'
                break
        
        if is_redundant: continue
            
        # D. Uniqueness Filter
        test_indicator = global_indicator.copy()
        safe_name = name if name not in test_indicator.columns else f"{name}_dup"
        test_indicator[safe_name] = cand['indicator'].iloc[:, 0]
        
        concurrency = test_indicator.sum(axis=1)
        u_t = test_indicator[safe_name] / concurrency 
        
        mask = test_indicator[safe_name] > 0
        uniq_vals = u_t[mask]
        
        if uniq_vals.empty:
            avg_uniq = 0.0
        else:
            avg_uniq = uniq_vals.mean() 
        
        if avg_uniq < tau_uniq:
            logger.debug(f"Discard {name}: Low Uniqueness ({avg_uniq:.2f})")
            cand['status'] = 'REJECT_LOW_UNIQ'
            continue
            
        # ACCEPT
        logger.info(f"Select  {name}: AUC={cand['auc']:.3f}, Uniq={avg_uniq:.2f}")
        cand['status'] = 'SELECTED'
        cand['final_uniqueness'] = avg_uniq
        
        geo = OutputGeometry(name, cand['family'], cand['events'], cand['labels'], 
                             cand['weights'], avg_uniq, cand['auc'])
        accepted_objects.append(geo)
        accepted_configs.append(cand)
        global_indicator[safe_name] = cand['indicator'].iloc[:, 0]
        
    return accepted_objects

# ==========================================
# 5. Main Orchestration
# ==========================================

def orthogonal_label_generation(
    price: pd.Series,
    volume: pd.Series,
    df_full: pd.DataFrame, 
    tau_auc: float = 0.55,
    tau_mi: float = 0.15,
    tau_uniq: float = 0.10
) -> List[OutputGeometry]:
    
    index = price.index
    
    # 0. Volatility for Dynamic Labeling & Floors
    daily_vol = price.pct_change().rolling(20).std()
    # Calculate robust floor for profitability (e.g. 25% of avg vol)
    avg_vol = daily_vol.mean()
    robust_floor = max(0.001, avg_vol * 0.25) if not np.isnan(avg_vol) else 0.002
    logger.info(f"Dynamic Label Floor Set to: {robust_floor:.5f}")
    
    # 1. Probe Features
    logger.info("--- Generating Probe Features (Basis Set) ---")
    X_probe = generate_probe_features(price, volume)
    
    # 2. Build 3D Hypothesis Grid
    regimes = [12, 24, 48]
    configs = []
    
    # --- CONTROLS ---
    configs.append({"f": "CONTROL", "t": "RANDOM", "g": RandomEvents(), "p": {"n_events": 200}})
    configs.append({"f": "CONTROL", "t": "TIME", "g": TimeEvents(), "p": {"step": 50}})
    
    # --- ANTI-BIAS ---
    configs.append({"f": "LOW_VOL", "t": "Q20", "g": LowVolatilityEvents(), "p": {"lookback": 50, "quantile": 0.20}})
    configs.append({"f": "CHOP", "t": "ER30", "g": ChopEvents(), "p": {"lookback": 20, "er_thresh": 0.3}})

    # --- STANDARD FAMILIES ---
    for r in regimes:
        # Volatility: Standard Z-Score AND Quantile Variants
        configs.append({"f": "VOL", "t": f"{r}_Z", "g": VolatilityShockEvents(), "p": {"lookback": r, "z": 2.0, "use_quantile": False}})
        configs.append({"f": "VOL", "t": f"{r}_Q", "g": VolatilityShockEvents(), "p": {"lookback": r, "use_quantile": True, "q": 0.95}})
        
        configs.append({"f": "MR", "t": str(r), "g": MeanReversionExtremeEvents(), "p": {"lookback": r, "z": 2.5}})
        configs.append({"f": "LIQ", "t": str(r), "g": LiquidityShockEvents(), "p": {"lookback": r, "z": 2.0}})
        
        # Breakouts: Split Long/Short for orthogonality
        configs.append({"f": "BREAK_L", "t": str(r), "g": BreakoutEvents(), "p": {"lookback": r, "side": "long"}})
        configs.append({"f": "BREAK_S", "t": str(r), "g": BreakoutEvents(), "p": {"lookback": r, "side": "short"}})
        
    trend_pairs = [(12, 24), (24, 48), (12, 48)]
    for s, l in trend_pairs:
        configs.append({"f": "TREND", "t": f"{s}_{l}", "g": TrendInitiationEvents(), "p": {"short": s, "long": l}})
        
    cusum_settings = [(12, 0.005), (24, 0.01), (48, 0.02)]
    for r, h in cusum_settings:
        configs.append({"f": "CUSUM_SYM", "t": str(r), "g": SymmetricCusumEvents(), "p": {"h": h}})
    
    # Improved CUSUM split into Orthogonal Components
    configs.append({"f": "CUSUM_IMP_TREND", "t": "STD_T", "g": ImprovedCUSUMEvents(), "p": {"k": 0.12, "w_trend": 1.0, "w_reversal": 0.0}})
    configs.append({"f": "CUSUM_IMP_REV", "t": "STD_R", "g": ImprovedCUSUMEvents(), "p": {"k": 0.12, "w_trend": 0.0, "w_reversal": 1.0}})
    
    for r in regimes:
        configs.append({"f": "HURST", "t": str(r), "g": HurstStateEvents(), "p": {"lookback": r * 2, "threshold": 0.6}})

    horizons = [12, 24, 48]
    candidates = []
    
    logger.info(f"--- Generating Candidates from {len(configs)} Generators ---")
    
    # Store detailed metrics for reporting
    metrics_report = []

    for conf in configs:
        fam, tag, gen, params = conf['f'], conf['t'], conf['g'], conf['p']
        
        if "CUSUM_IMP" in fam: data_src = df_full
        elif fam == "LIQ": data_src = volume
        else: data_src = price
            
        try:
            events = gen.generate(data_src, **params)
        except Exception as e:
            logger.warning(f"Generator {fam}_{tag} failed: {e}")
            continue
            
        n_events_generated = len(events)

        if n_events_generated < 30:
            metrics_report.append({
                'name': f"{fam}_{tag}",
                'stage': 'Generation',
                'n_events': n_events_generated,
                'status': 'Skipped (<30 events)'
            })
            continue
            
        for h in horizons:
            # 1. Dynamic MAE/MFE
            name_mae = f"{fam}_{tag}_MAE_H{h}"
            res_mae = dynamic_mae_mfe_label(
                price, events, 
                volatility=daily_vol, 
                horizon=h, 
                min_ret_factor=0.5, 
                min_ret_floor=robust_floor,
                dominance_ratio=1.5
            )
            
            # 2. Symmetric Version
            name_sym = f"{fam}_{tag}_SYM_H{h}"
            res_sym = vol_scaled_fixed_label(price, events, horizon=h, vol_lookback=20, z_threshold=1.5)
            
            for name, res in [(name_mae, res_mae), (name_sym, res_sym)]:
                if res.empty:
                    metrics_report.append({
                        'name': name,
                        'stage': 'Labeling',
                        'n_labeled': 0,
                        'status': 'Empty Labels'
                    })
                    continue
                
                y_cand = res['label']
                w_cand = res['weight']
                valid_idx = y_cand.index
                n_labeled = len(y_cand)

                # --- NEW: Class Balance Check ---
                balance_stats = check_class_balance(y_cand, min_class_samples=20)
                if not balance_stats['valid']:
                     metrics_report.append({
                        'name': name,
                        'stage': 'BalanceCheck',
                        'n_labeled': n_labeled,
                        'balance': balance_stats,
                        'status': 'Imbalanced'
                    })
                     continue
                # --------------------------------
                
                # Purged Probe
                X_curr = X_probe.loc[valid_idx]
                try:
                    auc_score = get_purged_lgbm_auc(X_curr, y_cand, w_cand, horizon_bars=h)
                except:
                    auc_score = 0.5
                    
                candidates.append({
                    "name": name,
                    "family": fam,
                    "events": events,
                    "labels": y_cand,
                    "weights": w_cand,
                    "auc": auc_score,
                    # Pass horizon to build accurate duration-based indicator matrix
                    "indicator": build_indicator_matrix(events, index, horizon=h),
                    "n_generated": n_events_generated,
                    "n_labeled": n_labeled,
                    "stats": balance_stats # Store stats
                })

                metrics_report.append({
                    'name': name,
                    'stage': 'Probe',
                    'n_generated': n_events_generated,
                    'n_labeled': n_labeled,
                    'auc': auc_score,
                    'status': 'Candidate'
                })

    # 4. Selection
    final_geometries = select_best_geometries(
        candidates, 
        tau_auc=tau_auc, 
        tau_mi=tau_mi, 
        tau_uniq=tau_uniq
    )
    
    # Final Report Log
    logger.info("\n=== Geometry Selection Report ===")

    # Update metrics_report with selection status
    # Create a map from name to status in candidates
    cand_status_map = {c['name']: c.get('status', 'Unknown') for c in candidates}

    report_df = pd.DataFrame(metrics_report)
    if not report_df.empty:
        # Update status from selection phase if applicable
        report_df['final_status'] = report_df['name'].map(cand_status_map).fillna(report_df['status'])

        # Sort by AUC descending (if available)
        if 'auc' in report_df.columns:
            report_df = report_df.sort_values(by='auc', ascending=False, na_position='last')

        logger.info(f"\n{report_df.to_string(index=False)}")

        # Summary by Family
        if 'family' not in report_df.columns:
            # Try to extract family from name
             report_df['family'] = report_df['name'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'Unknown')

        summary = report_df.groupby(['family', 'final_status']).size().unstack(fill_value=0)
        logger.info(f"\nSummary by Family:\n{summary.to_string()}")

    return final_geometries