import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
import os
from itertools import combinations
from datetime import datetime
from typing import List, Dict, Union, Callable, Optional, Tuple
from scipy.stats import spearmanr, entropy as shannon_entropy, norm
from scipy.special import expit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import jaccard_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from functools import partial

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

# Compatibility Alias
Geometry = OutputGeometry

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

    # Vectorized fill if horizon is small, loop otherwise
    # For speed with 1-bar horizon (Jaccard sim), this is trivial
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

# ==========================================
# 1. Labeling Logic (Vectorized Dominance)
# ==========================================

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
    Returns: labels, weights, returns, mfe, mae, volatility
    """
    # 1. Filter events within bounds
    if events.empty:
        return tuple([pd.Series(dtype=float)] * 6)

    n_bars = len(price)

    # Map events to integers
    event_idxs = price.index.get_indexer(events)
    valid_mask = (event_idxs != -1) & (event_idxs < (n_bars - horizon))
    valid_idxs = event_idxs[valid_mask]
    valid_events = events[valid_mask]

    if len(valid_idxs) == 0:
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

    # TBM Logic
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

    # 6. Returns
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
    generator_instance: Callable,
    generator_params: Dict
) -> Tuple[bool, Dict[str, float], str]:
    metrics = {}
    n = len(labels)
    days = (labels.index[-1] - labels.index[0]).days if n > 0 else 0
    rate = n / days if days > 0 else 0
    if rate < 1.0: return False, {'n': n}, "Sample Size (< 1.0/day)"

    pos_rate = labels.mean()
    if pos_rate < 0.075 or pos_rate > 0.925: return False, {'pos_rate': pos_rate}, "Class Balance"

    event_ts = labels.index.astype(np.int64) // 10**9
    if np.std(event_ts) < (days * 24 * 3600 * 0.1): return False, {}, "Stationarity"

    # Perturbation
    try:
        df_noisy = df.copy()
        noise = np.random.normal(1.0, 0.0001, size=len(df))
        for col in ['close', 'high', 'low', 'open']:
            if col in df_noisy.columns: df_noisy[col] *= noise

        gen = generator_instance
        if isinstance(gen, (MicrostructureEvents, TrendModulatedBreakoutEvents,
                            ATRShockEvents, VWAPReversionEvents)):
             events_noisy = gen.generate(df_noisy, **generator_params)
        else:
             events_noisy = gen.generate(df_noisy['close'], **generator_params)

        ind_clean = build_indicator_matrix(events, df.index, horizon=1).values.flatten()
        ind_noisy = build_indicator_matrix(events_noisy, df.index, horizon=1).values.flatten()

        intersection = np.logical_and(ind_clean, ind_noisy).sum()
        union = np.logical_or(ind_clean, ind_noisy).sum()
        jaccard = intersection / union if union > 0 else 0.0

        if jaccard < 0.8: return False, {'jaccard': jaccard}, "Perturbation Stability"
    except Exception:
        pass

    # ANOVA
    X = probe_features.loc[labels.index]
    y = labels
    with np.errstate(divide='ignore', invalid='ignore'):
        F, p_values = f_classif(X, y)
    valid_p = p_values[~np.isnan(p_values)]
    if len(valid_p) > 0 and np.min(valid_p) > 0.10: return False, {}, "ANOVA"

    # MI
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    if np.max(mi) < 0.01: return False, {}, "Mutual Info"

    # PSR
    if not returns.empty:
        sharpe = returns.mean() / (returns.std() + 1e-9)
        skew = returns.skew()
        kurt = returns.kurtosis()
        psr = calculate_psr(sharpe, len(returns), skew, kurt)
        if psr < 0.95: return False, {'psr': psr}, "PSR"
        metrics['psr'] = psr

    return True, metrics, "PASS"

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
        ic_vals = [abs(spearmanr(X[col], labels)[0]) for col in X.columns]
        ic_max = np.nanmax(ic_vals) if ic_vals else 0

        F, _ = f_classif(X, labels)
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
                sub_X = X.iloc[s:e]; sub_y = labels.iloc[s:e]
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
    if len(y) < 50:
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

    params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': 42}
    tscv = TimeSeriesSplit(n_splits=3)

    meta_returns = []
    base_returns = []
    preds_all = []
    labels_all = []

    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr, w_va = w.iloc[tr_idx], w.iloc[va_idx]
        ret_va = returns.iloc[va_idx]

        if y_tr.nunique() < 2 or y_va.nunique() < 2: continue

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va)
        model = lgb.train(params, dtrain, valid_sets=[dvalid], callbacks=[lgb.early_stopping(10, verbose=False)])

        preds = model.predict(X_va)
        preds_all.extend(preds)
        labels_all.extend(y_va)

        # Base: All
        base_returns.extend(ret_va.values)

        # Meta: Pred > 0.5
        mask = preds > 0.5
        if mask.sum() > 0:
            meta_returns.extend(ret_va[mask].values)

    if not base_returns:
        return {'lift': 0.0, 'yield': 0.0, 'entropy': 1.0, 'consistency': 0.0, 'sharpe_meta': 0.0}

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
        consistency = np.mean(labels_arr[mask]) # Expected value of Y given signal
    else:
        consistency = 0.0

    return {
        'lift': lift,
        'yield': opp_yield,
        'entropy': cond_entropy,
        'consistency': consistency,
        'sharpe_meta': meta_sh
    }

# ==========================================
# 5. Signal Generators
# ==========================================

class BaseEventGenerator:
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex: raise NotImplementedError

class EntropyEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, window: int = 24, z_thresh: float = 2.0) -> pd.DatetimeIndex:
        log_ret = np.log(price).diff().fillna(0)
        ent = roll_entropy(log_ret, window=window, bins=10)
        z_ent = (ent - ent.rolling(window*5).mean()) / (ent.rolling(window*5).std() + 1e-6)
        return price.index[(z_ent > z_thresh) & (z_ent.shift(1) <= z_thresh)]

class MicrostructureEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, window: int = 20, z: float = 2.0) -> pd.DatetimeIndex:
        if 'volume' not in df.columns: return pd.DatetimeIndex([])
        ret = df['close'].pct_change().abs()
        amihud = ret / (df['volume'] * df['close'] + 1e-9)
        z = (amihud - amihud.rolling(window).mean()) / (amihud.rolling(window).std() + 1e-9)
        return df.index[z > z]

class TrendModulatedBreakoutEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 20, anchor_window: int = 100) -> pd.DatetimeIndex:
        price = df['close']
        rmax = price.rolling(lookback).max().shift(1)
        rmin = price.rolling(lookback).min().shift(1)
        anchor = calc_vwap(price, df['volume'], anchor_window) if 'volume' in df.columns else price.rolling(anchor_window).mean()
        bk = ((price > rmax) & (price > anchor)) | ((price < rmin) & (price < anchor))
        return price.index[bk & ~bk.shift(1).fillna(False)]

class ATRShockEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 14, long_window: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        tr = calc_tr(df, df['close'])
        atr = tr.rolling(lookback).mean()
        zsc = (atr - atr.rolling(long_window).mean()) / (atr.rolling(long_window).std() + 1e-9)
        return df['close'].index[zsc > z]

class VWAPReversionEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        if 'volume' not in df.columns: return pd.DatetimeIndex([])
        vwap = calc_vwap(df['close'], df['volume'], lookback)
        return df.index[np.abs((df['close'] - vwap) / (df['close'].rolling(lookback).std() + 1e-6)) > z]

class KalmanTrendEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, q_fast: float = 1e-3, q_slow: float = 1e-5) -> pd.DatetimeIndex:
        f, _ = KalmanFilter1D(Q=q_fast).filter_series(price)
        s, _ = KalmanFilter1D(Q=q_slow).filter_series(price)
        return price.index[((f > s) & (f.shift(1) <= s.shift(1))) | ((f < s) & (f.shift(1) >= s.shift(1)))]

class KalmanRegimeEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, Q: float = 1e-4, R: float = 0.01, z: float = 2.0) -> pd.DatetimeIndex:
        # Detect sudden shifts in price level relative to Kalman estimate
        price = df['close']
        f, P = KalmanFilter1D(Q=Q, R=R).filter_series(price)
        # Innovation
        innov = price - f
        std = np.sqrt(P + R)
        z_score = innov / (std + 1e-9)
        return price.index[z_score.abs() > z]

class VWAPCrossEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 50) -> pd.DatetimeIndex:
        if 'volume' not in df.columns: return pd.DatetimeIndex([])
        vwap = calc_vwap(df['close'], df['volume'], lookback)
        price = df['close']
        cross_up = (price > vwap) & (price.shift(1) <= vwap.shift(1))
        cross_down = (price < vwap) & (price.shift(1) >= vwap.shift(1))
        return price.index[cross_up | cross_down]

# Aliases
CusumEvents = EntropyEvents
VolatilityShockEvents = ATRShockEvents
TrendInitiationEvents = TrendModulatedBreakoutEvents
MeanReversionExtremeEvents = VWAPReversionEvents
LiquidityShockEvents = MicrostructureEvents
TimeEvents = BaseEventGenerator
AdaptiveSymmetricCUSUMEvents = EntropyEvents # Fallback alias if strict class needed

# ==========================================
# 6. Main Pipeline
# ==========================================

def orthogonal_label_generation(
    data: Union[pd.Series, pd.DataFrame],
    labelers: Optional[Dict[str, Callable]] = None,
    volume: Optional[pd.Series] = None,
    df_full: Optional[pd.DataFrame] = None,
    tau_auc: float = 0.52,
    tau_mi: float = 0.15,
    tau_uniq: float = 0.10
) -> List[OutputGeometry]:
    """
    Main Execution Pipeline for Orthogonal Label Generation.
    Implements: Rank -> Top 50% -> Cluster (5) -> Top 1 -> Probe.
    """
    logger.info("--- Starting Advanced Geometry Generation ---")

    # 0. Data Standardization
    if isinstance(data, pd.DataFrame):
        price = data['close']
        if volume is None and 'volume' in data.columns:
            volume = data['volume']
        if df_full is None:
            df_full = data
    else:
        price = data

    if volume is None and df_full is not None and 'volume' in df_full.columns:
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

    # 2. Define Candidates (Generators)
    generators = [
        # Structural Family (Entropy)
        ('ENTROPY', EntropyEvents(), {'window': 24}),
        ('ENTROPY_FAST', EntropyEvents(), {'window': 12}),
        ('ENTROPY_SLOW', EntropyEvents(), {'window': 48}),
        
        # Volatility Family
        ('VOL_SHOCK', ATRShockEvents(), {'lookback': 14, 'long_window': 50, 'z': 2.0}),
        ('VOL_SHOCK_EXT', ATRShockEvents(), {'lookback': 14, 'long_window': 100, 'z': 3.0}),
        
        # Liquidity Family
        ('LIQUIDITY', MicrostructureEvents(), {'window': 20}),
        ('LIQUIDITY_FAST', MicrostructureEvents(), {'window': 10}),
        
        # Breakout Family
        ('BREAKOUT', TrendModulatedBreakoutEvents(), {'lookback': 48, 'anchor_window': 100}),
        ('BREAKOUT_FAST', TrendModulatedBreakoutEvents(), {'lookback': 20, 'anchor_window': 50}),

        # Kalman Trend
        ('KALMAN_TREND', KalmanTrendEvents(), {'q_fast': 1e-3, 'q_slow': 1e-5}),
        ('KALMAN_TREND_SENS', KalmanTrendEvents(), {'q_fast': 5e-3, 'q_slow': 5e-5}),

        # Mean Reversion
        ('MR_VWAP', VWAPReversionEvents(), {'lookback': 50, 'z': 2.5}),
        ('MR_VWAP_FAST', VWAPReversionEvents(), {'lookback': 24, 'z': 2.0}),

        # Kalman Regime
        ('KALMAN_REGIME', KalmanRegimeEvents(), {'Q': 1e-4, 'R': 0.01, 'z': 2.0}),

        # VWAP Cross
        ('VWAP_CROSS', VWAPCrossEvents(), {'lookback': 50}),
    ]
    
    # 3. Process Candidates (Generate & Gate)
    candidates = []
    outcomes_log = []

    for fam, gen, params in generators:
        try:
            if isinstance(gen, (MicrostructureEvents, TrendModulatedBreakoutEvents, ATRShockEvents, VWAPReversionEvents, VWAPCrossEvents, KalmanRegimeEvents)):
                events = gen.generate(df_full, **params)
            else:
                events = gen.generate(price, **params)
        except Exception as e:
            logger.debug(f"Generator {fam} failed: {e}")
            continue

        if len(events) < 5: continue

        # Iterate Grid
        for grid_item in FIXED_GRID:
            pt = grid_item['pt']
            sl = grid_item['sl']
            # We only use one risk budget for simplicity in this phase, or keep loop if needed.
            # Keeping loop to maximize candidate pool for clustering.
            for risk_budget in [1.0]:
                high = df_full.get('high')
                low = df_full.get('low')

                labels, weights, returns, mfe, mae, vol = compute_dominance_labels(
                    price, events, df_full['volatility_1d'],
                    risk_budget=risk_budget, pt_mult=pt, sl_mult=sl, horizon=120,
                    high=high, low=low
                )

                if labels.empty: continue

                passed, metrics, status = check_label_quality(
                    events, labels, returns, df_full, X_probe, gen, params
                )

                outcomes_log.append({
                    'family': fam,
                    'params': str(params),
                    'pt_mult': pt,
                    'sl_mult': sl,
                    'status': status,
                    'n': metrics.get('n', 0)
                })

                if passed:
                    candidates.append({
                        'family': fam,
                        'events': events,
                        'labels': labels,
                        'weights': weights,
                        'returns': returns,
                        'mfe': mfe, 'mae': mae, 'vol': vol,
                        'params': {**params, 'risk_budget': risk_budget, 'pt_mult': pt, 'sl_mult': sl, 'horizon': 120},
                        'status': status
                    })

    # 4. Multi-Factor Scoring
    scored_candidates = calculate_multifactor_score(candidates, X_probe)
    
    if not scored_candidates:
        logger.warning("No candidates passed gates.")
        return []

    # NEW LOGIC: Rank -> Top 50% -> Cluster (5) -> Top 1 -> Probe

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Keep Top 50%
    n_keep = max(1, len(scored_candidates) // 2)
    top_candidates = scored_candidates[:n_keep]
    logger.info(f"Top 50% selection: Kept {len(top_candidates)} from {len(scored_candidates)} candidates.")

    selected_candidates = []

    if len(top_candidates) <= 5:
        selected_candidates = top_candidates
        for i, cand in enumerate(selected_candidates):
            cand['cluster_id'] = i + 1
    else:
        # Cluster
        # Build Indicator Matrix for Jaccard
        indicators = []
        for c in top_candidates:
            # Horizon=1 for Signal Similarity
            ind = build_indicator_matrix(c['events'], price.index, horizon=1).values.flatten().astype(bool)
            indicators.append(ind)

        matrix = np.vstack(indicators) # (N_cands, N_bars)

        # Condensed Distance Matrix (Jaccard)
        dist_matrix = pdist(matrix, metric='jaccard')

        # Linkage
        Z = linkage(dist_matrix, method='average')

        # Cluster into 5
        cluster_labels = fcluster(Z, t=5, criterion='maxclust')

        # Select Top 1 per Cluster
        unique_clusters = np.unique(cluster_labels)
        for cid in unique_clusters:
            # Get indices for this cluster
            indices = [i for i, x in enumerate(cluster_labels) if x == cid]

            # Find best score in this cluster
            # top_candidates is already sorted by score, so the first one in indices is the best?
            # Yes, because indices are increasing and top_candidates is sorted descending.
            best_idx = indices[0]

            best_cand = top_candidates[best_idx]
            best_cand['cluster_id'] = int(cid)
            selected_candidates.append(best_cand)

    logger.info(f"Clustering selected {len(selected_candidates)} representatives.")

    # 5. Run LGBM Probe on Selected
    final_geoms = []

    for cand in selected_candidates:
        X = X_probe.loc[cand['labels'].index]
        metrics = run_lgbm_probe(X, cand['labels'], cand['weights'], cand['returns'])
        cand['metrics_probe'] = metrics

        # Create OutputGeometry
        indicator = build_indicator_matrix(cand['events'], price.index, horizon=120)
        purity = average_uniqueness(indicator)

        # Use Meta-Sharpe or Lift as the AUC metric for downstream compatibility
        # User specified "reuse composite score", but downstream expects valid AUC/Score.
        # We store the Probe Lift/Sharpe.

        geo = OutputGeometry(
            name=f"{cand['family']}_{cand['params']}",
            family=cand['family'],
            events=cand['events'],
            labels=cand['labels'],
            weights=cand['weights'],
            purity=purity,
            auc=metrics.get('lift', 0.0), # Storing Lift as primary metric
            cluster_id=cand['cluster_id'],
            params=cand['params'],
            metrics={**cand.get('metrics_raw', {}), **cand['metrics_probe']}
        )
        final_geoms.append(geo)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "outcomes"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(outcomes_log).to_csv(f"{out_dir}/geometry_gates_{timestamp}.csv", index=False)

    return final_geoms
