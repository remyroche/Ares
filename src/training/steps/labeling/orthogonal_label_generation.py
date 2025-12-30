import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable
from functools import partial

# Placeholder for existing codebase import
try:
    from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
except ImportError:
    generate_primary_signals = None

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

# ==========================================
# 1. Event Generators (The 7 Families + Controls)
# ==========================================

class BaseEventGenerator:
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

# --- CONTROL GROUPS (NULL HYPOTHESES) ---
class RandomEvents(BaseEventGenerator):
    """
    Null Hypothesis 1: Random Sampling.
    If this has High AUC, your model is overfitting or leaking.
    """
    def generate(self, price: pd.Series, n_events: int = 100) -> pd.DatetimeIndex:
        # Sample random indices from the price index
        if len(price) < n_events: n_events = len(price)
        # Use numpy choice for speed
        rng = np.random.default_rng(42) # Fixed seed for reproducibility check
        random_indices = rng.choice(price.index, size=n_events, replace=False)
        return pd.DatetimeIndex(np.sort(random_indices))

class TimeEvents(BaseEventGenerator):
    """
    Null Hypothesis 2: Time-based sampling.
    Tests if "Time alone" explains labels.
    """
    def generate(self, price: pd.Series, step: int = 50) -> pd.DatetimeIndex:
        return price.index[::step]

# --- ANTI-BIAS FAMILIES (REGIME BALANCE) ---
class LowVolatilityEvents(BaseEventGenerator):
    """
    Triggers when volatility is exceptionally LOW (Bottom Quantile).
    Ensures model learns 'Boring' regimes, not just crisis regimes.
    """
    def generate(self, price: pd.Series, lookback: int = 50, quantile: float = 0.20) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        # Rolling quantile to adapt to changing market baselines
        thresh = vol.rolling(lookback * 5).quantile(quantile)
        
        trigger = (vol < thresh) & (vol.shift(1) >= thresh.shift(1))
        return price.index[trigger]

class ChopEvents(BaseEventGenerator):
    """
    Triggers in Trendless/Choppy markets.
    Uses Efficiency Ratio (ER) < Threshold.
    """
    def generate(self, price: pd.Series, lookback: int = 20, er_thresh: float = 0.3) -> pd.DatetimeIndex:
        change = price.diff(lookback).abs()
        path = price.diff().abs().rolling(lookback).sum()
        er = change / (path + 1e-6)
        
        # Trigger when market becomes inefficient/choppy
        trigger = (er < er_thresh) & (er.shift(1) >= er_thresh)
        return price.index[trigger]

# --- STANDARD FAMILIES ---
class VolatilityShockEvents(BaseEventGenerator):
    """
    Supports both Z-Score (Parametric) and Quantile (Non-Parametric) triggers.
    """
    def generate(self, price: pd.Series, lookback: int = 50, z: float = 2.0, use_quantile: bool = False, q: float = 0.95) -> pd.DatetimeIndex:
        returns = price.pct_change()
        vol = returns.rolling(lookback).std()
        
        if use_quantile:
            # Non-parametric trigger
            thresh = vol.rolling(lookback*5).quantile(q)
            trigger = vol > thresh
            return price.index[trigger]
        else:
            # Standard Z-score trigger
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
    def generate(self, price: pd.Series, lookback: int = 20) -> pd.DatetimeIndex:
        rolling_max = price.rolling(lookback).max().shift(1)
        rolling_min = price.rolling(lookback).min().shift(1)
        breakout_high = price > rolling_max
        breakout_low = price < rolling_min
        return price.index[(breakout_high & ~breakout_high.shift(1).fillna(False)) | 
                           (breakout_low & ~breakout_low.shift(1).fillna(False))]

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
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        k = params.get('k', 0.12)
        if generate_primary_signals is not None:
            try:
                signals = generate_primary_signals(df, k=k)
                if 'consensus' in signals.columns:
                    return signals.index[signals['consensus'] != 0]
                return signals.index
            except Exception:
                pass
        try:
            return SymmetricCusumEvents().generate(df['close'], h=0.01)
        except:
            return pd.DatetimeIndex([])

class HurstStateEvents(BaseEventGenerator):
    def _get_hurst_exponent(self, ts):
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def generate(self, price: pd.Series, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        hurst = price.rolling(lookback, step=5).apply(self._get_hurst_exponent, raw=True)
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

def generate_probe_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Standard 'Basis Set' for Probe.
    Note: For true subspace orthogonality, features should theoretically be partitioned,
    but we use a global set here to benchmark all families on the same playing field.
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

def get_purged_lgbm_auc(X, y, w, horizon_bars=48) -> float:
    if len(y) < 50: return 0.5
    
    n_splits = 3
    gap = horizon_bars + 5 
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'auc_mu',
        'verbosity': -1,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.1,
        'n_estimators': 50
    }
    
    y_map = y.map({-1:0, 0:1, 1:2})
    
    if len(y) < 100: 
        split_idx = int(len(y) * 0.7)
        tr_idx, va_idx = np.arange(split_idx), np.arange(split_idx + gap, len(y))
        splits = [(tr_idx, va_idx)]
    else:
        splits = tscv.split(X)
    
    valid_splits_count = 0
    
    for tr_idx, va_idx in splits:
        if len(tr_idx) < 20 or len(va_idx) < 20: continue
        
        curr_X_tr, curr_X_va = X.iloc[tr_idx], X.iloc[va_idx]
        curr_y_tr, curr_y_va = y_map.iloc[tr_idx], y_map.iloc[va_idx]
        curr_w_tr, curr_w_va = w.iloc[tr_idx], w.iloc[va_idx]
        
        dtrain = lgb.Dataset(curr_X_tr, label=curr_y_tr, weight=curr_w_tr)
        dvalid = lgb.Dataset(curr_X_va, label=curr_y_va, weight=curr_w_va)
        
        model = lgb.train(params, dtrain, valid_sets=[dvalid], 
                          callbacks=[lgb.early_stopping(10, verbose=False)])
        
        try:
            score = model.best_score['valid_0']['auc_mu']
            scores.append(score)
            valid_splits_count += 1
        except:
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
    
    accepted_configs = []
    accepted_objects = []
    global_indicator = pd.DataFrame() 
    
    print(f"--- Starting Selection on {len(candidates)} Candidates ---")
    
    for cand in candidates:
        name = cand['name']
        
        # --- NULL HYPOTHESIS CHECK ---
        # If Controls are high-ranking, warn the user heavily.
        if cand['family'] == 'CONTROL':
            if cand['auc'] > 0.54: # Threshold for "Suspiciously Learnable"
                print(f"⚠️  WARNING: Control Geometry {name} has High AUC ({cand['auc']:.3f}). Possible Leakage!")
            # We explicitly do NOT accept controls into the final set, they are for audit.
            continue
            
        # A. Junk Filter
        if cand['auc'] < tau_auc:
            continue

        # B. Stability Filter
        if not label_distribution_stable(cand['labels']):
            print(f"Discard {name}: Unstable Labels")
            continue
            
        # C. Redundancy Filter
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
    df_full: pd.DataFrame, 
    tau_auc: float = 0.55,
    tau_mi: float = 0.15,
    tau_uniq: float = 0.10
) -> List[OutputGeometry]:
    
    index = price.index
    
    # 0. Volatility for Dynamic Labeling
    daily_vol = price.pct_change().rolling(20).std()
    
    # 1. Probe Features
    print("--- Generating Probe Features (Basis Set) ---")
    X_probe = generate_probe_features(price, volume)
    
    # 2. Build 3D Hypothesis Grid
    regimes = [12, 24, 48]
    configs = []
    
    # --- CONTROLS (NULL HYPOTHESES) ---
    configs.append({"f": "CONTROL", "t": "RANDOM", "g": RandomEvents(), "p": {"n_events": 200}})
    configs.append({"f": "CONTROL", "t": "TIME", "g": TimeEvents(), "p": {"step": 50}})
    
    # --- ANTI-BIAS (BALANCING) ---
    configs.append({"f": "LOW_VOL", "t": "Q20", "g": LowVolatilityEvents(), "p": {"lookback": 50, "quantile": 0.20}})
    configs.append({"f": "CHOP", "t": "ER30", "g": ChopEvents(), "p": {"lookback": 20, "er_thresh": 0.3}})

    # --- STANDARD FAMILIES ---
    for r in regimes:
        # Note: Volatility now uses quantile trigger option? Let's use standard Z for main grid
        configs.append({"f": "VOL", "t": str(r), "g": VolatilityShockEvents(), "p": {"lookback": r, "z": 2.0}})
        configs.append({"f": "MR", "t": str(r), "g": MeanReversionExtremeEvents(), "p": {"lookback": r, "z": 2.5}})
        configs.append({"f": "LIQ", "t": str(r), "g": LiquidityShockEvents(), "p": {"lookback": r, "z": 2.0}})
        configs.append({"f": "BREAK", "t": str(r), "g": BreakoutEvents(), "p": {"lookback": r}})
        
    trend_pairs = [(12, 24), (24, 48), (12, 48)]
    for s, l in trend_pairs:
        configs.append({"f": "TREND", "t": f"{s}_{l}", "g": TrendInitiationEvents(), "p": {"short": s, "long": l}})
        
    cusum_settings = [(12, 0.005), (24, 0.01), (48, 0.02)]
    for r, h in cusum_settings:
        configs.append({"f": "CUSUM_SYM", "t": str(r), "g": SymmetricCusumEvents(), "p": {"h": h}})
    
    configs.append({"f": "CUSUM_IMP", "t": "STD", "g": ImprovedCUSUMEvents(), "p": {"k": 0.12}})
    
    for r in regimes:
        configs.append({"f": "HURST", "t": str(r), "g": HurstStateEvents(), "p": {"lookback": r * 2, "threshold": 0.6}})

    horizons = [12, 24, 48]
    candidates = []
    
    print(f"--- Generating Candidates from {len(configs)} Generators ---")
    
    for conf in configs:
        fam, tag, gen, params = conf['f'], conf['t'], conf['g'], conf['p']
        
        if fam == "CUSUM_IMP": data_src = df_full
        elif fam == "LIQ": data_src = volume
        else: data_src = price
            
        try:
            events = gen.generate(data_src, **params)
        except Exception:
            continue
            
        if len(events) < 30: continue
            
        for h in horizons:
            # 1. Dynamic MAE/MFE
            name_mae = f"{fam}_{tag}_MAE_H{h}"
            res_mae = dynamic_mae_mfe_label(
                price, events, 
                volatility=daily_vol, 
                horizon=h, 
                min_ret_factor=0.5,   
                dominance_ratio=1.5
            )
            
            # 2. Symmetric Version
            name_sym = f"{fam}_{tag}_SYM_H{h}"
            res_sym = vol_scaled_fixed_label(price, events, horizon=h, vol_lookback=20, z_threshold=1.5)
            
            for name, res in [(name_mae, res_mae), (name_sym, res_sym)]:
                if res.empty: continue
                
                y_cand = res['label']
                w_cand = res['weight']
                valid_idx = y_cand.index
                
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
