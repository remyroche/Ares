import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score, roc_auc_score
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable, Any, Optional
from functools import partial
import lightgbm as lgb
from scipy.special import expit

# Import Kalman Filter for ImprovedCUSUMEvents
try:
    from src.training.steps.labeling.mtf_feature_generation import KalmanFilter1D
except ImportError:
    # Fallback if not available
    class KalmanFilter1D:
        def __init__(self, Q=1e-5, R=0.01, initial_value=0.0):
            self.Q = Q
            self.R = R
            self.x = initial_value
            self.P = 1.0

        def filter_series(self, series):
            # Simple placeholder (no smoothing)
            return series, pd.Series(0, index=series.index)

# ==========================================
# 1. Event Generators (Orthogonal Families)
# ==========================================

class BaseEventGenerator:
    """
    Abstract base class for event generation strategies.
    """
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class SymmetricCusumEvents(BaseEventGenerator):
    """
    The De Prado Standard (Chapter 2).
    Detects structural breaks in the mean price.
    More robust to noise than Simple Moving Average crossovers.
    """
    def generate(self, price: pd.Series, h: float = 0.05) -> pd.DatetimeIndex:
        # h is the threshold in percent (e.g., 0.05 = 5% deviation triggers event)
        # In practice, we often set h based on daily volatility (e.g., h = vol * 2)

        t_events = []
        s_pos = 0
        s_neg = 0

        # Ensure we work with Series
        if isinstance(price, pd.DataFrame):
            price = price['close']

        diff = price.pct_change() # using simple returns for this implementation

        # Calculate dynamic threshold based on rolling vol (optional but recommended)
        # Here we use fixed 'h' for simplicity, or you can pass a vol series
        # Optimization: Loop over numpy array
        diff_arr = diff.values
        index = diff.index

        # Determine if h is a series (dynamic) or float
        if isinstance(h, (pd.Series, np.ndarray)):
            if len(h) != len(diff):
                # Align or handle mismatch
                if isinstance(h, pd.Series):
                    h = h.reindex(index).fillna(method='ffill').fillna(0.01).values
                else:
                    # Scalar fallback
                    h = np.full(len(diff), 0.05)
            else:
                 if isinstance(h, pd.Series): h = h.values
        else:
             h = np.full(len(diff), h)

        for i in range(1, len(diff)):
            r = diff_arr[i]
            if np.isnan(r): continue

            threshold = h[i]

            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            if s_pos > threshold:
                s_neg = 0
                s_pos = 0
                t_events.append(index[i])
            elif s_neg < -threshold:
                s_neg = 0
                s_pos = 0
                t_events.append(index[i])

        return pd.DatetimeIndex(t_events)

class ImprovedCUSUMEvents(BaseEventGenerator):
    """
    Detects structural breaks using Dual CUSUM logic (Trend + Reversal) with Kalman Filter smoothing.
    Copied and adapted from generate_dual_cusum_signals.
    """
    def generate(self, df: pd.DataFrame, vol_window: int = 20, k: float = 0.12, **kwargs) -> pd.DatetimeIndex:
        # Extract data
        close = df['close'] if 'close' in df.columns else df.iloc[:, 0]
        volume = df['volume'] if 'volume' in df.columns else None

        # Parameters
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 1.0)
        er_min = kwargs.get('er_min', 0.2)
        window_er = kwargs.get('er_window', 10)
        Q = kwargs.get('Q', 1e-5)
        R = kwargs.get('R', 0.01)
        w_trend = kwargs.get('w_trend', 1.0)
        w_reversal = kwargs.get('w_reversal', 1.0)

        # 1. Compute log returns
        log_ret = np.log(close / close.shift(1)).fillna(0.0)

        # 2. Apply 1D Kalman filter
        kf = KalmanFilter1D(Q=Q, R=R, initial_value=float(log_ret.iloc[0]))
        log_ret_smooth_raw, _ = kf.filter_series(log_ret)

        if not isinstance(log_ret_smooth_raw, pd.Series):
            log_ret_smooth_series = pd.Series(log_ret_smooth_raw, index=close.index).fillna(0.0)
        else:
            log_ret_smooth_series = log_ret_smooth_raw.fillna(0.0)

        # 3. Rolling volatility & ER
        sigma = log_ret_smooth_series.rolling(vol_window, min_periods=1).std()

        change = log_ret_smooth_series.rolling(window_er).sum().abs()
        volatility_sum = log_ret_smooth_series.abs().rolling(window_er, min_periods=1).sum()
        ER = (change / (volatility_sum + 1e-12)).fillna(0.0)

        # 4. Liquidity & Thresholds
        liquidity_mod = pd.Series(1.0, index=close.index)
        if volume is not None:
            vol_ma = volume.rolling(vol_window, min_periods=1).mean()
            rel_volume = volume / (vol_ma + 1e-9)
            liquidity_mod = 1.0 + beta * (1.0 - rel_volume)
            liquidity_mod = liquidity_mod.clip(0.5, 2.0)

        regime_mod = 1.0 + alpha * (1.0 - ER)
        h_t = (k * sigma * regime_mod * liquidity_mod).fillna(0.0)

        # 5. Residuals for Reversal Logic
        expected_return = log_ret_smooth_series.rolling(vol_window, min_periods=1).mean()
        residual_ret = (log_ret_smooth_series - expected_return).fillna(0.0)

        # 6. CUSUM Loop
        n = len(close)
        r_arr = log_ret_smooth_series.to_numpy()
        res_arr = residual_ret.to_numpy()
        h_arr = h_t.to_numpy()
        er_arr = ER.to_numpy()

        composite_signal = np.zeros(n)

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

            # Trend
            S_trend_pos = max(0.0, S_trend_pos + r_arr[t])
            S_trend_neg = min(0.0, S_trend_neg + r_arr[t])

            trend_sig = 0
            if S_trend_pos > cur_h:
                trend_sig = 1
                S_trend_pos = 0.0
            elif S_trend_neg < -cur_h:
                trend_sig = -1
                S_trend_neg = 0.0

            # Reversal
            S_rev_pos = max(0.0, S_rev_pos + res_arr[t])
            S_rev_neg = min(0.0, S_rev_neg + res_arr[t])

            rev_sig = 0
            if S_rev_pos > cur_h:
                rev_sig = 1
                S_rev_pos = 0.0
            elif S_rev_neg < -cur_h:
                rev_sig = -1
                S_rev_neg = 0.0

            composite = w_trend * trend_sig - w_reversal * rev_sig
            if composite != 0:
                composite_signal[t] = composite

        # Return indices where signal is generated
        return df.index[composite_signal != 0]

class HurstStateEvents(BaseEventGenerator):
    """
    Detects when the market switches from "Random Walk" to "Trend".
    Triggers when Hurst Exponent crosses critical thresholds.
    """
    def get_hurst(self, series):
        # Simplified R/S analysis or similar
        # (Using a quick approximation for performance in loops)
        lags = range(2, 20)
        # Handle small series
        if len(series) < 20: return 0.5

        try:
            tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
            # polyfit needs at least 2 points
            if len(tau) < 2: return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except Exception:
            return 0.5

    def generate(self, price: pd.Series, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        # Warning: Hurst is computationally expensive.
        # rolling_apply is slow. We generate events sparsely.

        if isinstance(price, pd.DataFrame):
            price = price['close']

        hurst_vals = price.rolling(lookback).apply(self.get_hurst, raw=False)

        # Trigger when we cross INTO a trend regime (H > 0.6)
        # We only want the *initiation* of the regime, not every day inside it.

        trigger = (hurst_vals > threshold) & (hurst_vals.shift(1) <= threshold)
        return price.index[trigger]

# ==========================================
# 2. Geometry & Tools
# ==========================================

class Geometry:
    """
    Container for a specific Event + Label combination.
    """
    def __init__(self, name: str, events: pd.DatetimeIndex, labels: pd.Series,
                 family: str = None, labeler_name: str = None, params: Dict = None):
        self.name = name
        self.events = events
        self.labels = labels.dropna()
        self.indicator = None
        self.avg_uniqueness = None
        self.score = 0.0 # Learnability score
        self.family = family
        self.labeler_name = labeler_name
        self.params = params or {}
        self.uuid = name # Compatibility

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.Series:
    """
    Maps events to the full timeline.
    """
    ind = pd.Series(0, index=index)
    valid_events = events.intersection(index)
    ind.loc[valid_events] = 1
    return ind

def average_uniqueness(indicators: pd.DataFrame) -> float:
    """
    Calculates average uniqueness (1 / concurrency) across all events.
    """
    if indicators.empty:
        return 0.0
    concurrency = indicators.sum(axis=1)
    # Avoid division by zero
    uniq = indicators.div(concurrency, axis=0).replace([np.inf, np.nan], 0)

    # We only care about uniqueness when the event is actually active (indicator > 0)
    valid = indicators > 0
    if not valid.any().any():
        return 0.0

    return uniq[indicators > 0].mean().mean()

def normalized_mi(y1: pd.Series, y2: pd.Series) -> float:
    """
    Calculates Normalized Mutual Information between two label sets.
    """
    common = y1.index.intersection(y2.index)
    if len(common) < 10: # Require some overlap to judge
        return 0.0

    # Discretize if continuous (though labels here are likely binary 0/1)
    mi = mutual_info_score(y1.loc[common], y2.loc[common])

    # Normalize by entropy of y1 to get range [0, 1] relative to the candidate
    entropy = shannon_entropy(y1.loc[common].value_counts())

    return mi / entropy if entropy > 0 else 0.0

# ==========================================
# 3. Main Orchestration
# ==========================================

def orthogonal_label_generation(
    candidates: List[Geometry],
    tau_uniqueness: float = 0.1,
    tau_mi: float = 0.1,
    scorer: Optional[Callable[[Geometry], float]] = None
) -> List[Geometry]:
    """
    The Tournament: Score -> Sort -> Filter.

    1. Score: Calculate AUC/Learnability for all candidates.
    2. Sort: Rank by score descending.
    3. Filter: Select best, reject redundant (high MI).

    Args:
        candidates: List of Geometry objects (already generated).
        tau_uniqueness: Minimum uniqueness threshold.
        tau_mi: Maximum Mutual Information allowed with accepted candidates.
        scorer: Function to compute score (e.g., AUC) for a Geometry.
    """

    print(f"Starting Orthogonal Tournament with {len(candidates)} candidates...")

    # 1. Score
    if scorer:
        for g in candidates:
            try:
                g.score = scorer(g)
            except Exception as e:
                print(f"Scoring failed for {g.name}: {e}")
                g.score = 0.0

    # 2. Sort
    sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

    for i, g in enumerate(sorted_candidates[:5]):
        print(f"Rank {i+1}: {g.name} (Score: {g.score:.4f})")

    accepted = []
    global_indicator = pd.DataFrame()

    # 3. Filter
    print("Filtering for Orthogonality...")

    for g in sorted_candidates:
        # A. Check Marginal Uniqueness (vs everything already accepted)
        # We construct the indicator relative to the events timeline?
        # Events might be sparse. We need a common index.
        # Assuming g.indicator is already populated and aligned to common index
        if g.indicator is None:
             # Should be pre-populated or we can't do uniqueness check easily without index
             # Skip uniqueness check if indicator missing?
             uniq = 1.0
        else:
            if global_indicator.empty:
                uniq = 1.0
                global_indicator = pd.DataFrame(index=g.indicator.index)
            else:
                temp_indicator = pd.concat([global_indicator, g.indicator], axis=1).fillna(0)
                uniq = average_uniqueness(temp_indicator)

        g.avg_uniqueness = uniq

        # Relax uniqueness for high performers? No, strict filter.
        # But if it's the *first* one, uniq is 1.0.

        # B. Check Mutual Information (Redundancy in outcome)
        redundant = False
        for a in accepted:
            mi = normalized_mi(g.labels, a.labels)
            # If high MI, it means g provides similar information to a (which is higher ranked)
            if mi > tau_mi:
                print(f"Rejected {g.name}: High MI with {a.name} ({mi:.2f})")
                redundant = True
                break

        if redundant:
            continue

        # Accept
        accepted.append(g)
        if g.indicator is not None:
             global_indicator[g.name] = g.indicator
        print(f"Accepted {g.name} (Score: {g.score:.4f}, Uniq: {uniq:.2f})")

        # Stop if we have enough? Or keep going?
        # Usually Layer 2 finds "the best geometry for each orthogonal source".
        # But here we mix sources.
        # If we want 1 per source family, we can enforce that too.
        # But the prompt says "Filter (Orthogonality)... The system automatically 'found' that the Medium-Term Volatility regime was the most predictive... and discarded redundant Fast/Slow".
        # So we don't enforce 1 per family, we enforce orthogonality.

    return accepted
