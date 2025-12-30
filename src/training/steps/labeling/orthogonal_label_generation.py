import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score, roc_auc_score
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable, Any, Tuple
from functools import partial

# Import CUSUM generator from codebase
try:
    from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
except ImportError:
    pass

# ==========================================
# 1. Event Generators (Orthogonal Families)
# ==========================================

class BaseEventGenerator:
    """Abstract base class."""
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class VolatilityShockEvents(BaseEventGenerator):
    """Detects sudden spikes in volatility (Expanding Window)."""
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        if 'volatility_1d' in df.columns:
            vol = df['volatility_1d']
        else:
            vol = df['close'].pct_change().rolling(lookback).std()

        vol_mean = vol.expanding(min_periods=lookback).mean()
        vol_std = vol.expanding(min_periods=lookback).std()
        vol_std = vol_std.replace(0, np.nan)

        zscore = (vol - vol_mean) / vol_std
        return df.index[zscore > z]

class TrendInitiationEvents(BaseEventGenerator):
    """Detects moving average crossovers."""
    def generate(self, df: pd.DataFrame, short: int = 20, long: int = 100) -> pd.DatetimeIndex:
        price = df['close']
        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()

        cross_up = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
        cross_down = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

        return df.index[cross_up | cross_down]

class MeanReversionExtremeEvents(BaseEventGenerator):
    """Detects significant deviation from local mean."""
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        price = df['close']
        mean = price.rolling(lookback).mean()
        std = price.rolling(lookback).std()
        zscore = (price - mean) / std
        return df.index[np.abs(zscore) > z]

class LiquidityShockEvents(BaseEventGenerator):
    """Detects volume spikes."""
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        volume = df['volume']
        vol_mean = volume.expanding(min_periods=lookback).mean()
        vol_std = volume.expanding(min_periods=lookback).std()
        vol_std = vol_std.replace(0, np.nan)
        zscore = (volume - vol_mean) / vol_std
        return df.index[zscore > z]

class TimeEvents(BaseEventGenerator):
    """Clock-based sampling."""
    def generate(self, df: pd.DataFrame, step: int = 50) -> pd.DatetimeIndex:
        return df.index[::step]

class ImprovedCUSUMEvents(BaseEventGenerator):
    """Wrapper for existing CUSUM filter."""
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        k = params.get('k', 0.12)
        try:
            signals = generate_primary_signals(df, k=k)
            if 'consensus' in signals.columns:
                return signals.index[signals['consensus'] != 0]
        except Exception:
            pass
        return pd.DatetimeIndex([])

class SymmetricCusumEvents(BaseEventGenerator):
    """
    De Prado Standard Symmetric CUSUM.
    Detects structural breaks in mean price.
    """
    def generate(self, df: pd.DataFrame, h: float = 0.05) -> pd.DatetimeIndex:
        # h is threshold in percent
        # Use simple returns
        price = df['close']
        diff = price.pct_change()

        t_events = []
        s_pos = 0.0
        s_neg = 0.0

        # Iterate (using simple loop for clarity/correctness, can be numba-optimized)
        # Assuming diff index is monotonic

        # Use numpy arrays for speed
        r_arr = diff.values
        idx_arr = diff.index

        # Skip first NaN
        for i in range(1, len(r_arr)):
            r = r_arr[i]
            if np.isnan(r): continue

            s_pos = max(0.0, s_pos + r)
            s_neg = min(0.0, s_neg + r)

            if s_pos > h:
                s_neg = 0.0
                s_pos = 0.0
                t_events.append(idx_arr[i])
            elif s_neg < -h:
                s_neg = 0.0
                s_pos = 0.0
                t_events.append(idx_arr[i])

        return pd.DatetimeIndex(t_events)

class HurstStateEvents(BaseEventGenerator):
    """
    Trigger when Hurst Exponent initiates a Trend Regime (> 0.6).
    """
    def _get_hurst_rolling_apply(self, x):
        # Simplified R/S or similar approximation
        # x is a numpy array of price
        try:
            lags = np.arange(2, 20)
            # Calculate standard deviation of diffs for various lags
            # This is a rough proxy for diffusion speed
            tau = []
            for lag in lags:
                # diff(lag)
                d = x[lag:] - x[:-lag]
                if len(d) < 2:
                    tau.append(np.nan)
                    continue
                tau.append(np.std(d))

            # Polyfit log-log
            # tau ~ lag^H
            # log(tau) ~ H * log(lag)
            valid = np.isfinite(tau) & (np.array(tau) > 0)
            if np.sum(valid) < 3: return 0.5

            log_lags = np.log(lags[valid])
            log_tau = np.log(np.array(tau)[valid])

            poly = np.polyfit(log_lags, log_tau, 1)
            return poly[0] # This slope is H
        except:
            return 0.5

    def generate(self, df: pd.DataFrame, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        price = df['close']
        # Use pandas rolling with a simplified hurst calc
        # rolling().apply is slow, so we might want to step or optimize.
        # Given constraints, we apply it.

        hurst = price.rolling(lookback).apply(self._get_hurst_rolling_apply, raw=True)

        # Trigger initiation
        trigger = (hurst > threshold) & (hurst.shift(1) <= threshold)
        return df.index[trigger]

# ==========================================
# 2. Geometry & Tools
# ==========================================

class Geometry:
    def __init__(self, name: str, events: pd.DatetimeIndex, labels: pd.Series,
                 family: str = None, labeler_name: str = None, params: Dict = None):
        self.name = name
        self.events = events
        self.labels = labels.dropna()
        self.indicator = None
        self.avg_uniqueness = None
        self.family = family
        self.labeler_name = labeler_name
        self.params = params or {}
        self.score = 0.0 # Learnability/AUC Score

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.Series:
    ind = pd.Series(0, index=index)
    valid_events = events.intersection(index)
    ind.loc[valid_events] = 1
    return ind

def average_uniqueness(indicators: pd.DataFrame) -> float:
    if indicators.empty: return 0.0
    concurrency = indicators.sum(axis=1)
    uniq = indicators.div(concurrency, axis=0).replace([np.inf, np.nan], 0)
    valid = indicators > 0
    if not valid.any().any(): return 0.0
    return uniq[indicators > 0].mean().mean()

def normalized_mi(y1: pd.Series, y2: pd.Series) -> float:
    common = y1.index.intersection(y2.index)
    if len(common) < 10: return 0.0
    mi = mutual_info_score(y1.loc[common], y2.loc[common])
    entropy = shannon_entropy(y1.loc[common].value_counts())
    return mi / entropy if entropy > 0 else 0.0

def label_distribution_stable(labels: pd.Series, splits: int = 5, eps: float = 0.1) -> bool:
    if len(labels) < splits * 20: return True
    chunks = np.array_split(labels, splits)
    for a, b in combinations(chunks, 2):
        if len(a) < 10 or len(b) < 10: continue
        pa = a.value_counts(normalize=True)
        pb = b.value_counts(normalize=True)
        pa, pb = pa.align(pb, fill_value=0)
        d = shannon_entropy(pa, pb)
        if not np.isfinite(d): d = 1.0
        if d > eps: return False
    return True

# ==========================================
# 3. Main Orchestration
# ==========================================

def orthogonal_label_generation(
    df: pd.DataFrame,
    labelers: Dict[str, Callable[[pd.DataFrame, pd.DatetimeIndex], pd.Series]],
    scorer: Callable[[pd.DatetimeIndex, pd.Series], float] = None,
    tau_uniqueness: float = 0.1,
    tau_mi: float = 0.1
) -> List[Geometry]:

    index = df.index
    candidates = []

    # 1. Define Event Families & Variations
    # Instantiate generators
    gen_vol = VolatilityShockEvents()
    gen_trend = TrendInitiationEvents()
    gen_mr = MeanReversionExtremeEvents()
    gen_liq = LiquidityShockEvents()
    gen_cusum = ImprovedCUSUMEvents()
    gen_sym = SymmetricCusumEvents()
    gen_hurst = HurstStateEvents()
    gen_time = TimeEvents()

    # Define Variations
    # Format: (FamilyName, Generator, ParamsDict)
    variations = [
        # Volatility
        ("VOL_FAST", gen_vol, {"lookback": 20, "z": 2.0}),
        ("VOL_MED", gen_vol, {"lookback": 50, "z": 2.0}),
        ("VOL_SLOW", gen_vol, {"lookback": 100, "z": 2.0}),

        # Trend
        ("TREND_FAST", gen_trend, {"short": 10, "long": 50}),
        ("TREND_MED", gen_trend, {"short": 20, "long": 100}),
        ("TREND_SLOW", gen_trend, {"short": 50, "long": 200}),

        # Mean Reversion
        ("MR_FAST", gen_mr, {"lookback": 20, "z": 2.5}),
        ("MR_MED", gen_mr, {"lookback": 50, "z": 2.5}),
        ("MR_SLOW", gen_mr, {"lookback": 100, "z": 2.5}),

        # Liquidity
        ("LIQ_FAST", gen_liq, {"lookback": 20}),
        ("LIQ_MED", gen_liq, {"lookback": 50}),

        # CUSUM
        ("CUSUM_STD", gen_cusum, {"k": 0.12}),
        ("CUSUM_SENS", gen_cusum, {"k": 0.05}),

        # Symmetric CUSUM
        ("SYM_CUSUM_05", gen_sym, {"h": 0.05}),
        ("SYM_CUSUM_02", gen_sym, {"h": 0.02}),

        # Hurst
        ("HURST_INIT", gen_hurst, {"lookback": 100, "threshold": 0.6}),

        # Time
        ("TIME", gen_time, {"step": 50})
    ]

    print(f"Generating candidates from {len(variations)} variations x {len(labelers)} labelers...")

    # 2. Generate All Candidates
    for fam_name, generator, fam_params in variations:
        try:
            # Generate Events
            events = generator.generate(df, **fam_params)

            if len(events) < 15: # Minimum events
                continue

            for lbl_name, lbl_func in labelers.items():
                # Extract baked-in params if partial
                lbl_params = {}
                if isinstance(lbl_func, partial):
                    lbl_params = lbl_func.keywords

                # Combine params for record keeping
                full_params = {**fam_params, **lbl_params}

                # Generate Labels
                try:
                    labels = lbl_func(df, events)
                except Exception:
                    continue

                if labels.empty or labels.dropna().empty:
                    continue

                # Score Candidate (Learnability AUC)
                # If scorer provided, use it. Else default 0.5.
                score = 0.5
                if scorer:
                    try:
                        score = scorer(events, labels)
                    except Exception as e:
                        print(f"Scoring failed for {fam_name}_{lbl_name}: {e}")
                        score = 0.0

                if score < 0.5: # If worse than random, maybe flip? No, De Prado says 0.5 is floor.
                    score = 0.0

                g = Geometry(
                    name=f"{fam_name}_{lbl_name}",
                    events=events,
                    labels=labels,
                    family=fam_name,
                    labeler_name=lbl_name,
                    params=full_params
                )
                g.score = score
                g.indicator = build_indicator_matrix(events, index)
                candidates.append(g)

        except Exception as e:
            print(f"Error processing variation {fam_name}: {e}")
            continue

    # 3. Sort by Learnability Score (The Tournament)
    print(f"Sorting {len(candidates)} candidates by Learnability Score...")
    candidates.sort(key=lambda x: x.score, reverse=True)

    # Debug print top 5
    for i, c in enumerate(candidates[:5]):
        print(f"Rank {i+1}: {c.name} (Score={c.score:.4f}, Events={len(c.events)})")

    # 4. Filter for Uniqueness and Orthogonality
    accepted = []
    global_indicator = pd.DataFrame(index=index)

    print("Starting Orthogonality Filter...")

    for g in candidates:
        # A. Check Marginal Uniqueness
        if global_indicator.empty:
            uniq = 1.0
        else:
            temp_indicator = pd.concat([global_indicator, g.indicator], axis=1).fillna(0)
            uniq = average_uniqueness(temp_indicator)

        g.avg_uniqueness = uniq

        if uniq < tau_uniqueness:
            # print(f"Rejected {g.name}: Uniqueness {uniq:.2f} < {tau_uniqueness}")
            continue

        # B. Check Mutual Information (Redundancy in outcome)
        redundant = False
        for a in accepted:
            mi = normalized_mi(g.labels, a.labels)
            if mi > tau_mi:
                # print(f"Rejected {g.name}: High MI with {a.name} ({mi:.2f})")
                redundant = True
                break
        if redundant:
            continue

        # C. Check Stability
        if not label_distribution_stable(g.labels):
            # print(f"Rejected {g.name}: Unstable label distribution")
            continue

        # Accept
        accepted.append(g)
        global_indicator[g.name] = g.indicator
        print(f"Accepted {g.name} (Score={g.score:.4f})")

    return accepted
