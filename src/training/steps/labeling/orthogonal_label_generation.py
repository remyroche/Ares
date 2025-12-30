import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
import os
from itertools import combinations
from datetime import datetime
from typing import List, Dict, Union, Callable, Optional, Tuple
from joblib import Parallel, delayed
from scipy.stats import spearmanr, entropy as shannon_entropy, norm
from scipy.special import expit
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import jaccard_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# Setup Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==========================================
# 0. Data Structures & Helpers
# ==========================================

class OutputGeometry:
    """
    Standardized output object for the pipeline.
    """
    def __init__(self, name, family, events, labels, weights, purity, auc, params, metrics=None):
        self.name = name
        self.family = family
        self.events = events
        self.labels = labels
        self.weights = weights
        self.purity = purity      # Uniqueness Score
        self.auc = auc            # Probe AUC
        self.params = params
        self.metrics = metrics or {}
    
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
    for loc in event_locs:
        end_loc = min(loc + horizon, n_bars)
        arr[loc:end_loc] += 1
    
    arr = np.clip(arr, 0, 1)
    return pd.DataFrame(arr, index=index, columns=[0])

# ==========================================
# 1. Labeling Logic (MFE/MAE Dominance)
# ==========================================

def compute_dominance_labels(
    price: pd.Series,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    kappa: float = 2.0,
    sl_mult: float = 1.0,
    horizon: int = 120,
    transaction_cost: float = 0.003
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MFE/MAE Dominance Labeling.
    Label = 1 if MFE > Kappa * MAE (and not stopped out first).
    Returns: labels, weights, returns (estimated)
    """
    labels = {}
    weights = {}
    returns = {}
    
    vol_s = volatility.reindex(events).fillna(method='bfill').fillna(0.01)
    price_vals = price.values
    
    min_profit = transaction_cost * 1.1

    for t in events:
        if t not in price.index: continue

        idx_start = price.index.get_loc(t)
        idx_end = min(idx_start + horizon, len(price) - 1)
        if idx_start >= idx_end: continue

        path = price_vals[idx_start : idx_end + 1]
        path_ret = (path / path[0]) - 1

        stop_dist = max(0.004, vol_s[t] * sl_mult)

        mfe = np.max(path_ret)
        mae = np.max(-path_ret)

        stop_idxs = np.where(path_ret < -stop_dist)[0]
        first_stop = stop_idxs[0] if len(stop_idxs) > 0 else len(path_ret) + 1

        mfe_idx = np.argmax(path_ret)

        label = 0
        weight = 1.0
        ret_val = -stop_dist # Default to stop loss

        if mfe_idx < first_stop:
            if mfe > min_profit:
                mae_safe = max(mae, 1e-9)
                ratio = mfe / mae_safe
                if ratio > kappa:
                    label = 1
                    weight = np.log1p(ratio) * np.log1p(mfe / transaction_cost)
                    ret_val = mfe # Optimistic return for winner

        labels[t] = label
        weights[t] = weight
        returns[t] = ret_val

    if not labels:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    return pd.Series(labels), pd.Series(weights), pd.Series(returns)

# ==========================================
# 2. Quality Gates & Checks
# ==========================================

def calculate_psr(sharpe, n, skew, kurt, target_sharpe=0):
    """Probabilistic Sharpe Ratio"""
    if n < 2: return 0.0
    # Standard Error of Sharpe
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
    """
    Quality Gates including Stability and PSR.
    """
    metrics = {}

    # 1. Sample Size
    n = len(labels)
    days = (labels.index[-1] - labels.index[0]).days if n > 0 else 0
    rate = n / days if days > 0 else 0
    if rate < 3.0:
        return False, {'n': n, 'rate': rate}, "Sample Size (< 3/day)"

    # 2. Class Balance
    pos_rate = labels.mean()
    if pos_rate < 0.075 or pos_rate > 0.925:
        return False, {'pos_rate': pos_rate}, "Class Balance (< 7.5% minority)"

    # 3. Stationarity
    event_ts = labels.index.astype(np.int64) // 10**9
    if np.std(event_ts) < (days * 24 * 3600 * 0.1):
         q_counts = labels.index.to_series().groupby(pd.Grouper(freq='Q')).count()
         if (q_counts == 0).sum() > len(q_counts) * 0.5:
             return False, {'std_time': np.std(event_ts)}, "Stationarity (Clustered)"

    # 4. Perturbation Stability (Implemented)
    try:
        # Create noisy DataFrame (1bp noise)
        df_noisy = df.copy()
        noise = np.random.normal(1.0, 0.0001, size=len(df))
        # Apply noise to Price fields
        for col in ['close', 'high', 'low', 'open']:
            if col in df_noisy.columns:
                df_noisy[col] = df_noisy[col] * noise

        # Dispatch generation
        # Determine if generator needs df or price
        # This matches the logic in the main loop
        gen = generator_instance
        if isinstance(gen, (MicrostructureEvents, TrendModulatedBreakoutEvents,
                            ATRShockEvents, VWAPReversionEvents)):
             events_noisy = gen.generate(df_noisy, **generator_params)
        else:
             events_noisy = gen.generate(df_noisy['close'], **generator_params)

        # Jaccard Index
        # We need indicator overlap
        ind_clean = build_indicator_matrix(events, df.index, horizon=1).values.flatten()
        ind_noisy = build_indicator_matrix(events_noisy, df.index, horizon=1).values.flatten()

        # Fast Jaccard on binary
        # Intersection / Union
        intersection = np.logical_and(ind_clean, ind_noisy).sum()
        union = np.logical_or(ind_clean, ind_noisy).sum()

        jaccard = intersection / union if union > 0 else 0.0

        if jaccard < 0.8:
             return False, {'jaccard': jaccard}, "Perturbation Stability (< 0.8)"

    except Exception as e:
        logger.warning(f"Stability check failed: {e}")
        pass # Don't fail the pipeline if check errors out

    # 5. ANOVA "Sanity Check"
    X = probe_features.loc[labels.index]
    y = labels
    with np.errstate(divide='ignore', invalid='ignore'):
        F, p_values = f_classif(X, y)
    valid_p = p_values[~np.isnan(p_values)]
    if len(valid_p) > 0 and np.min(valid_p) > 0.10:
         return False, {'min_p_val': np.min(valid_p)}, "ANOVA (No Signal)"

    # 6. Mutual Information
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    if np.max(mi) < 0.01:
        return False, {'max_mi': np.max(mi)}, "Mutual Info (< 0.01)"

    # 7. Probabilistic Sharpe Ratio (PSR) (Implemented)
    if not returns.empty:
        sharpe = returns.mean() / (returns.std() + 1e-9)
        skew = returns.skew()
        kurt = returns.kurtosis()
        psr = calculate_psr(sharpe, len(returns), skew, kurt)

        if psr < 0.95:
             return False, {'psr': psr}, "PSR (< 0.95)"
        metrics['psr'] = psr
    
    metrics.update({
        'n': n,
        'pos_rate': pos_rate,
        'min_p_val': np.min(valid_p) if len(valid_p) > 0 else 1.0,
        'max_mi': np.max(mi),
        'jaccard': jaccard if 'jaccard' in locals() else 1.0
    })
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
        events = cand['events']
        labels = cand['labels']
        n = len(labels)

        X = probe_features.loc[labels.index]
        ic_vals = [abs(spearmanr(X[col], labels)[0]) for col in X.columns]
        ic_max = np.nanmax(ic_vals) if ic_vals else 0

        F, _ = f_classif(X, labels)
        f_max = np.nanmax(F) if len(F) > 0 else 0

        significance = np.log1p(n)

        chunk_size = n // 3
        if chunk_size > 10:
            ic_chunks = []
            for i in range(3):
                s = i * chunk_size
                e = (i + 1) * chunk_size if i < 2 else n
                sub_X = X.iloc[s:e]
                sub_y = labels.iloc[s:e]
                chunk_ics = [abs(spearmanr(sub_X[col], sub_y)[0]) for col in sub_X.columns]
                ic_chunks.append(np.nanmax(chunk_ics))
            stability = 1.0 / (np.std(ic_chunks) + 1e-6)
        else:
            stability = 0.5
            
        counts = labels.value_counts(normalize=True)
        balance = shannon_entropy(counts)
        
        indicator = build_indicator_matrix(events, X.index, horizon=cand['params']['horizon'])
        density = average_uniqueness(indicator)

        cand['metrics_raw'] = {
            'ic': ic_max,
            'f_stat': f_max,
            'significance': significance,
            'stability': stability,
            'balance': balance,
            'density': density
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
            row['density']
        )
        cand['score'] = final_score
        cand['power'] = power

    return scores

# ==========================================
# 4. Probe (LGBM)
# ==========================================

def run_lgbm_probe(X, y, w) -> float:
    if len(y) < 50: return 0.5
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.1,
        'num_leaves': 8,
        'min_child_samples': 10,
        'seed': 42
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        w_tr, w_va = w.iloc[tr_idx], w.iloc[va_idx]
        if y_tr.nunique() < 2 or y_va.nunique() < 2: continue
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va)
        model = lgb.train(params, dtrain, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(10, verbose=False)])
        scores.append(model.best_score['valid_0']['auc'])
    return np.mean(scores) if scores else 0.5

# ==========================================
# 5. Signal Generators (Expanded)
# ==========================================

class BaseEventGenerator:
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class EntropyEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, window: int = 24, z_thresh: float = 2.0) -> pd.DatetimeIndex:
        log_ret = np.log(price).diff().fillna(0)
        ent = roll_entropy(log_ret, window=window, bins=10)
        ent_mean = ent.rolling(window*5).mean()
        ent_std = ent.rolling(window*5).std()
        z_ent = (ent - ent_mean) / (ent_std + 1e-6)
        trigger = (z_ent > z_thresh) & (z_ent.shift(1) <= z_thresh)
        return price.index[trigger]

class MicrostructureEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, window: int = 20, z: float = 2.0) -> pd.DatetimeIndex:
        if 'volume' not in df.columns: return pd.DatetimeIndex([])
        ret = df['close'].pct_change().abs()
        amihud = ret / (df['volume'] * df['close'] + 1e-9)
        z_score = (amihud - amihud.rolling(window).mean()) / (amihud.rolling(window).std() + 1e-9)
        return df.index[z_score > z]

class TrendModulatedBreakoutEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 20, anchor_window: int = 100) -> pd.DatetimeIndex:
        price = df['close']
        rolling_max = price.rolling(lookback).max().shift(1)
        rolling_min = price.rolling(lookback).min().shift(1)
        if 'volume' in df.columns:
            anchor = calc_vwap(price, df['volume'], anchor_window)
        else:
            anchor = price.rolling(anchor_window).mean()
        breakout = ((price > rolling_max) & (price > anchor)) | ((price < rolling_min) & (price < anchor))
        return price.index[breakout & ~breakout.shift(1).fillna(False)]

class ATRShockEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 14, long_window: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        tr = calc_tr(df, df['close'])
        atr = tr.rolling(lookback).mean()
        z_score = (atr - atr.rolling(long_window).mean()) / (atr.rolling(long_window).std() + 1e-9)
        return df['close'].index[z_score > z]

class VWAPReversionEvents(BaseEventGenerator):
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        if 'volume' not in df.columns: return pd.DatetimeIndex([])
        vwap = calc_vwap(df['close'], df['volume'], lookback)
        zscore = (df['close'] - vwap) / (df['close'].rolling(lookback).std() + 1e-6)
        return df.index[np.abs(zscore) > z]

class KalmanTrendEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, q_fast: float = 1e-3, q_slow: float = 1e-5) -> pd.DatetimeIndex:
        f, _ = KalmanFilter1D(Q=q_fast).filter_series(price)
        s, _ = KalmanFilter1D(Q=q_slow).filter_series(price)
        cross = ((f > s) & (f.shift(1) <= s.shift(1))) | ((f < s) & (f.shift(1) >= s.shift(1)))
        return price.index[cross]

class CusumEvents(BaseEventGenerator):
    def generate(self, price: pd.Series, **kwargs):
        return pd.DatetimeIndex([])

CusumEvents = EntropyEvents
VolatilityShockEvents = ATRShockEvents
TrendInitiationEvents = TrendModulatedBreakoutEvents
MeanReversionExtremeEvents = VWAPReversionEvents
LiquidityShockEvents = MicrostructureEvents
TimeEvents = BaseEventGenerator

# ==========================================
# 6. Main Pipeline
# ==========================================

def orthogonal_label_generation(
    df: pd.DataFrame,
    *args, **kwargs
) -> List[OutputGeometry]:
    
    logger.info("--- Starting Multi-Factor Orthogonal Geometry Generation ---")
    
    price = df['close']
    
    probe_features = pd.DataFrame(index=df.index)
    probe_features['ret_1'] = price.pct_change()
    probe_features['vol_20'] = probe_features['ret_1'].rolling(20).std()
    probe_features['rsi_14'] = 100 - (100 / (1 + (price.diff().where(lambda x: x>0, 0).rolling(14).mean() / price.diff().where(lambda x: x<0, 0).abs().rolling(14).mean().replace(0, 1e-9))))
    probe_features.fillna(0, inplace=True)

    tp_sl_configs = [
        (1.5, 1.0), (1.5, 2.0),
        (2.0, 1.0), (2.0, 2.0),
        (3.0, 1.0),
        (4.0, 1.0)
    ]
    lookbacks = [20, 30, 40, 50]
    
    generators = []
    for w in lookbacks:
        generators.append(('ENTROPY', EntropyEvents(), {'window': w}))
        generators.append(('LIQUIDITY', MicrostructureEvents(), {'window': w}))
        generators.append(('BREAKOUT', TrendModulatedBreakoutEvents(), {'lookback': w, 'anchor_window': w*4}))
        generators.append(('VOL_SHOCK', ATRShockEvents(), {'lookback': max(10, w//2), 'long_window': w}))
        generators.append(('MR_VWAP', VWAPReversionEvents(), {'lookback': w}))
        generators.append(('KALMAN', KalmanTrendEvents(), {'q_fast': 1/w, 'q_slow': 1/(w*5)}))

    candidates = []
    outcomes_log = []
    
    for fam, gen, params in generators:
        try:
            if isinstance(gen, (MicrostructureEvents, TrendModulatedBreakoutEvents,
                                ATRShockEvents, VWAPReversionEvents)):
                events = gen.generate(df, **params)
            else:
                events = gen.generate(price, **params)
        except Exception:
            continue
            
        if len(events) < 5: continue
        
        for (kappa, sl_mult) in tp_sl_configs:
            labels, weights, returns = compute_dominance_labels(
                price, events, df['volatility_1d'],
                kappa=kappa, sl_mult=sl_mult, horizon=120
            )

            if labels.empty: continue

            passed, metrics, status = check_label_quality(
                events, labels, returns, df, probe_features, gen, params
            )

            row = {
                'family': fam,
                'params': str(params),
                'kappa': kappa,
                'sl_mult': sl_mult,
                'status': status,
                'n': metrics.get('n', 0),
                'score': 0
            }
            outcomes_log.append(row)

            if passed:
                candidates.append({
                    'family': fam,
                    'events': events,
                    'labels': labels,
                    'weights': weights,
                    'params': {**params, 'kappa': kappa, 'sl_mult': sl_mult, 'horizon': 120},
                    'status': status
                })

    scored_candidates = calculate_multifactor_score(candidates, probe_features)
    final_geoms = []
    families = set(c['family'] for c in scored_candidates)
    
    for f in families:
        fam_cands = [c for c in scored_candidates if c['family'] == f]
        fam_cands.sort(key=lambda x: x.get('score', 0), reverse=True)
        top5 = fam_cands[:5]
        
        best_cand = None
        best_auc = -1
        
        for cand in top5:
            X = probe_features.loc[cand['labels'].index]
            auc = run_lgbm_probe(X, cand['labels'], cand['weights'])
            cand['auc'] = auc
            if auc > best_auc:
                best_auc = auc
                best_cand = cand
        
        if best_cand and best_auc > 0.51:
            indicator = build_indicator_matrix(best_cand['events'], df.index, horizon=120)
            purity = average_uniqueness(indicator)
            
            geo = OutputGeometry(
                name=f"{f}_{best_cand['params']}",
                family=f,
                events=best_cand['events'],
                labels=best_cand['labels'],
                weights=best_cand['weights'],
                purity=purity,
                auc=best_auc,
                params=best_cand['params'],
                metrics=best_cand.get('metrics_raw', {})
            )
            final_geoms.append(geo)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "outcomes"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(outcomes_log).to_csv(f"{out_dir}/geometry_gates_{timestamp}.csv", index=False)
    
    logger.info(f"Selected {len(final_geoms)} Top-1 geometries.")
    return final_geoms
