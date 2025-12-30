import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import KFold
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable, Optional
from functools import partial

from src.training.steps.labeling.mtf_feature_generation import KalmanFilter1D

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

# Alias for backward compatibility
Geometry = OutputGeometry

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Maps events to the full timeline as a binary indicator series.
    Returns DataFrame to be compatible with arithmetic operations later.
    """
    ind = pd.Series(0, index=index)
    valid_events = events.intersection(index)
    ind.loc[valid_events] = 1
    return ind.to_frame()

def average_uniqueness(indicators: pd.DataFrame) -> float:
    """
    Calculates average uniqueness (1 / concurrency) across all events.
    Matches AFML Ch. 4 logic exactly.
    """
    if indicators.empty:
        return 0.0

    concurrency = indicators.sum(axis=1)
    uniqueness = indicators.div(concurrency, axis=0)

    # only count rows where this geometry is active
    # (We are interested in the uniqueness of the SIGNALS, not the quiet periods)
    mask = indicators > 0
    # Use boolean indexing to get the values where mask is True.
    # Note: uniqueness[mask] returns a DataFrame with NaNs where mask is False.
    # We want the mean of the valid values.
    uniq_vals = uniqueness[mask]

    if uniq_vals.count().sum() == 0:
        return 0.0

    # Mean across columns (geometries), then mean across time (events)
    # This ensures equal weight to geometries in the score
    return uniq_vals.mean().mean()

def normalized_mi(y1: pd.Series, y2: pd.Series) -> float:
    """
    Calculates Symmetric Normalized Mutual Information (0 to 1).
    Uses min(H(X), H(Y)) as denominator to prevent bias against low-entropy signals.
    """
    common = y1.index.intersection(y2.index)
    if len(common) < 30: 
        return 0.0

    # Calculate MI on common indices
    mi = mutual_info_score(y1.loc[common], y2.loc[common])
    
    # Calculate Entropies
    h1 = shannon_entropy(y1.loc[common].value_counts())
    h2 = shannon_entropy(y2.loc[common].value_counts())
    
    # Symmetric normalization
    denom = min(h1, h2)
    return mi / denom if denom > 0 else 0.0

def label_distribution_stable(labels: pd.Series, splits: int = 5, eps: float = 0.15) -> bool:
    """
    Checks if label distribution is stationary across time chunks.
    Rejects geometries that are just "Lucky runs".
    """
    if len(labels) < splits * 10: 
        return True # Not enough data to fail check

    # Sort index to ensure temporal splitting
    labels = labels.sort_index()
    chunks = np.array_split(labels, splits)
    
    for a, b in combinations(chunks, 2):
        if len(a) < 10 or len(b) < 10:
            continue
            
        pa = a.value_counts(normalize=True)
        pb = b.value_counts(normalize=True)
        
        # Align indexes (ensure both have -1, 0, 1)
        pa, pb = pa.align(pb, fill_value=0)
        
        # Calculate Divergence (using Entropy diff as proxy for stability)
        d = shannon_entropy(pa, pb)
        if not np.isfinite(d): 
             d = 1.0
             
        if d > eps:
            return False
    return True

# ==========================================
# 1. Event Generators (The 6 Families)
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

    # 2. Apply 1D Kalman filter (Reuse existing optimized class)
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

class VolatilityShockEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        # Expanding window to avoid look-ahead bias
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
    """
    De Prado Standard (Chapter 2).
    """
    def generate(self, price: pd.Series, h: float = 0.01) -> pd.DatetimeIndex:
        t_events = []
        s_pos = 0
        s_neg = 0
        # Log returns for scale invariance
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

# Alias for backward compatibility
CusumEvents = SymmetricCusumEvents

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
                w_trend * dual_signals['trend_signal'] -
                w_reversal * dual_signals['reversal_signal']
            )

            return composite.index[composite != 0]

        except Exception as e:
            print(f"Improved CUSUM failed, falling back: {e}")
            # Fallback
            try:
                return SymmetricCusumEvents().generate(close, h=0.01)
            except:
                return pd.DatetimeIndex([])

class HurstStateEvents(BaseEventGenerator):
    """
    Detects transition to Trend Regime (H > 0.6).
    """
    def _get_hurst_exponent(self, ts):
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def generate(self, price: pd.Series, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        # Step optimization for speed
        hurst = price.rolling(lookback, step=5).apply(self._get_hurst_exponent, raw=True)
        hurst = hurst.reindex(price.index).ffill() 
        trigger = (hurst > threshold) & (hurst.shift(1) <= threshold)
        return price.index[trigger]

class TimeEvents(BaseEventGenerator):
    """
    Generates events based on time intervals (e.g. daily, hourly).
    Useful for time-based labeling strategies.
    """
    def generate(self, price: pd.Series, freq: str = '1D') -> pd.DatetimeIndex:
        # Resample to frequency and take the last timestamp
        return price.resample(freq).last().dropna().index

# ==========================================
# 2. Labeling Logic
# ==========================================

def mae_mfe_weighted_label(price: pd.Series, events: pd.DatetimeIndex, 
                           horizon: int = 24, 
                           min_return: float = 0.004, 
                           dominance_ratio: float = 1.5) -> pd.DataFrame:
    """
    Standard MAE/MFE Labeler. Bias: Trend following / Momentum.
    """
    results = {}
    price_arr = price.values
    
    for t in events:
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
        
        if mfe > min_return and long_ratio >= dominance_ratio:
            lbl = 1
            weight = np.log(1.0 + long_ratio)
            
        elif abs(mae) > min_return and short_ratio >= dominance_ratio:
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
    """
    Symmetric Labeler. Bias: None (Pure direction).
    Useful for Mean Reversion to avoid Trend bias of MAE/MFE.
    Target: Price change > Z * Volatility
    """
    results = {}
    returns = price.pct_change()
    vol = returns.rolling(vol_lookback).std()
    
    for t in events:
        if t not in price.index: continue
        t_idx = price.index.get_loc(t)
        if t_idx + horizon >= len(price): continue
        
        # Get volatility at entry
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
# 3. Probe & Validation Tools
# ==========================================

def generate_probe_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Generates the 'Basis Set' (6 features) for the Tournament.
    """
    df = pd.DataFrame(index=price.index)
    df['ret_1'] = np.log(price).diff(1)
    df['ret_12'] = np.log(price).diff(12) 
    
    vol_20 = df['ret_1'].rolling(20).std()
    vol_100 = df['ret_1'].rolling(100).std()
    df['vol_ratio'] = vol_20 / (vol_100 + 1e-6)
    
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ma_50 = price.rolling(50).mean()
    df['trend_dist'] = (price / ma_50) - 1
    
    vol_ma = volume.rolling(20).mean()
    df['vol_shock'] = volume / (vol_ma + 1e-6)
    
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def get_lgbm_auc(X, y, w) -> float:
    """
    Runs a fast LGBM Probe (Depth 3) to assess learnability.
    """
    if len(y) < 30: return 0.5
    
    kf = KFold(n_splits=3, shuffle=False)
    scores = []
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'auc_mu',
        'verbosity': -1,
        'max_depth': 3, # Constrained
        'num_leaves': 8,
        'learning_rate': 0.1,
        'n_estimators': 50
    }
    
    y_map = y.map({-1:0, 0:1, 1:2})
    
    for tr_idx, va_idx in kf.split(X):
        # Safety alignment
        curr_X_tr, curr_X_va = X.iloc[tr_idx], X.iloc[va_idx]
        curr_y_tr, curr_y_va = y_map.iloc[tr_idx], y_map.iloc[va_idx]
        curr_w_tr, curr_w_va = w.iloc[tr_idx], w.iloc[va_idx]
        
        dtrain = lgb.Dataset(curr_X_tr, label=curr_y_tr, weight=curr_w_tr)
        dvalid = lgb.Dataset(curr_X_va, label=curr_y_va, weight=curr_w_va)
        
        model = lgb.train(params, dtrain, valid_sets=[dvalid], 
                          callbacks=[lgb.early_stopping(10, verbose=False)])
        
        try:
            score = model.best_score['valid_0']['auc_mu']
        except:
            score = 0.5
        scores.append(score)
        
    return np.mean(scores)

# ==========================================
# 4. Selection Logic
# ==========================================

def select_best_geometries(candidates: List[Dict], tau_auc=0.55, tau_mi=0.15, tau_uniq=0.10) -> List[OutputGeometry]:
    """
    Applies the Selection Formula: Filter Junk -> Sort Quality -> Select Orthogonal.
    """
    # 1. SORT: Learnability first, then Richness
    candidates.sort(
        key=lambda x: (
            -x['auc'],          # Primary: Learnability
            -len(x['labels'])   # Secondary: Sample Richness
        )
    )
    
    accepted_configs = []
    accepted_objects = []
    global_indicator = pd.DataFrame() 
    
    print(f"--- Starting Selection on {len(candidates)} Candidates ---")
    
    for cand in candidates:
        name = cand['name']
        
        # A. Junk Filter (Quality)
        if cand['auc'] < tau_auc:
            # print(f"Discard {name}: Low AUC ({cand['auc']:.3f})")
            continue

        # B. Stability Filter
        if not label_distribution_stable(cand['labels']):
            print(f"Discard {name}: Unstable Labels")
            continue
            
        # C. Redundancy Filter (MI)
        is_redundant = False
        for acc in accepted_configs:
            mi_score = normalized_mi(cand['labels'], acc['labels'])
            if mi_score > tau_mi:
                print(f"Discard {name}: Redundant with {acc['name']} (MI={mi_score:.2f})")
                is_redundant = True
                break
        
        if is_redundant: continue
            
        # D. Uniqueness Filter
        test_indicator = global_indicator.copy()
        # Ensure column name is unique if reusing names (unlikely but safe)
        safe_name = name if name not in test_indicator.columns else f"{name}_dup"
        test_indicator[safe_name] = cand['indicator'].iloc[:, 0] # Extract series from DF
        
        concurrency = test_indicator.sum(axis=1)
        u_t = test_indicator[safe_name] / concurrency 
        
        mask = test_indicator[safe_name] > 0
        uniq_vals = u_t[mask]
        
        if uniq_vals.empty:
            avg_uniq = 0.0
        else:
            avg_uniq = uniq_vals.mean() # Mean of the candidate's active periods
        
        # Enforce uniqueness even for the first candidate
        if avg_uniq < tau_uniq:
            print(f"Discard {name}: Low Uniqueness ({avg_uniq:.2f})")
            continue
            
        # ACCEPT
        print(f"Select  {name}: AUC={cand['auc']:.3f}, Uniq={avg_uniq:.2f}")
        
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
    df_full: pd.DataFrame, # Required for CUSUM/Vol
    tau_auc: float = 0.55,
    tau_mi: float = 0.15,
    tau_uniq: float = 0.10
) -> List[OutputGeometry]:
    
    index = price.index
    
    # 1. Probe Features
    print("--- Generating Probe Features (Basis Set) ---")
    X_probe = generate_probe_features(price, volume)
    
    # 2. Build 3D Hypothesis Grid
    regimes = [12, 24, 48]
    configs = []
    
    # Grid Definitions
    for r in regimes:
        configs.append({"f": "VOL", "t": str(r), "g": VolatilityShockEvents(), "p": {"lookback": r, "z": 2.0}})
        configs.append({"f": "MR", "t": str(r), "g": MeanReversionExtremeEvents(), "p": {"lookback": r, "z": 2.5}})
        configs.append({"f": "LIQ", "t": str(r), "g": LiquidityShockEvents(), "p": {"lookback": r, "z": 2.0}})
        
    trend_pairs = [(12, 24), (24, 48), (12, 48)]
    for s, l in trend_pairs:
        configs.append({"f": "TREND", "t": f"{s}_{l}", "g": TrendInitiationEvents(), "p": {"short": s, "long": l}})
        
    cusum_settings = [(12, 0.005), (24, 0.01), (48, 0.02)]
    for r, h in cusum_settings:
        configs.append({"f": "CUSUM_SYM", "t": str(r), "g": SymmetricCusumEvents(), "p": {"h": h}})
    
    configs.append({"f": "CUSUM_IMP", "t": "STD", "g": ImprovedCUSUMEvents(), "p": {"k": 0.12}})
    
    for r in regimes:
        configs.append({"f": "HURST", "t": str(r), "g": HurstStateEvents(), "p": {"lookback": r * 2, "threshold": 0.6}})

    # 3. Labeler Factories (Horizons: 12, 24, 48)
    # Mixing MAE/MFE (Directional Dominance) and Vol-Scaled (Symmetric)
    horizons = [12, 24, 48]
    
    candidates = []
    print(f"--- Generating Candidates from {len(configs)} Generators ---")
    
    for conf in configs:
        fam, tag, gen, params = conf['f'], conf['t'], conf['g'], conf['p']
        
        # Select Data Source
        if fam == "CUSUM_IMP":
            data_src = df_full
        elif fam == "LIQ":
            data_src = volume
        else:
            data_src = price
            
        try:
            events = gen.generate(data_src, **params)
        except Exception:
            continue
            
        if len(events) < 30: continue
            
        for h in horizons:
            # 1. MAE/MFE Version (Trend Bias)
            name_mae = f"{fam}_{tag}_MAE_H{h}"
            res_mae = mae_mfe_weighted_label(price, events, horizon=h, min_return=0.004, dominance_ratio=1.5)
            
            # 2. Symmetric Version (Vol Bias)
            name_sym = f"{fam}_{tag}_SYM_H{h}"
            res_sym = vol_scaled_fixed_label(price, events, horizon=h, vol_lookback=20, z_threshold=1.5)
            
            # Process both
            for name, res in [(name_mae, res_mae), (name_sym, res_sym)]:
                if res.empty: continue
                
                y_cand = res['label']
                w_cand = res['weight']
                valid_idx = y_cand.index
                
                # Probe
                X_curr = X_probe.loc[valid_idx]
                try:
                    auc_score = get_lgbm_auc(X_curr, y_cand, w_cand)
                except:
                    auc_score = 0.5
                    
                candidates.append({
                    "name": name,
                    "family": fam,
                    "events": events,
                    "labels": y_cand,
                    "weights": w_cand,
                    "auc": auc_score,
                    "indicator": build_indicator_matrix(events, index)
                })

    # 4. Selection
    final_geometries = select_best_geometries(
        candidates, 
        tau_auc=tau_auc, 
        tau_mi=tau_mi, 
        tau_uniq=tau_uniq
    )
    
    return final_geometries
