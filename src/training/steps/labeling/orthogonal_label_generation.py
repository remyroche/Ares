import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable, Any
from functools import partial

# Import CUSUM generator
# We assume this is available in the codebase as per memory/context
try:
    from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
except ImportError:
    # Fallback or placeholder if direct import fails during test/planning, but should work in production
    pass

# ==========================================
# 1. Event Generators (Orthogonal Families)
# ==========================================

class BaseEventGenerator:
    """
    Abstract base class for event generation strategies.
    """
    def generate(self, data: Union[pd.Series, pd.DataFrame], **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class VolatilityShockEvents(BaseEventGenerator):
    """
    Detects sudden spikes in volatility relative to the running history.
    Uses expanding window to avoid look-ahead bias.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        # Expecting 'volatility_1d' or calculate it from close
        if 'volatility_1d' in df.columns:
            vol = df['volatility_1d']
        else:
            vol = df['close'].pct_change().rolling(lookback).std()

        # Expanding window statistics
        vol_mean = vol.expanding(min_periods=lookback).mean()
        vol_std = vol.expanding(min_periods=lookback).std()

        vol_std = vol_std.replace(0, np.nan)
        zscore = (vol - vol_mean) / vol_std

        return df.index[zscore > z]

class TrendInitiationEvents(BaseEventGenerator):
    """
    Detects moving average crossovers indicating a regime shift.
    """
    def generate(self, df: pd.DataFrame, short: int = 20, long: int = 100) -> pd.DatetimeIndex:
        price = df['close']
        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()

        # Signal: Short crosses above Long (and was previously below)
        # We capture both directions if needed, but here we check crossover
        # For simplicity, we can return both Up and Down crossovers
        cross_up = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
        cross_down = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

        return df.index[cross_up | cross_down]

class MeanReversionExtremeEvents(BaseEventGenerator):
    """
    Detects when price deviates significantly from its local mean.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.5) -> pd.DatetimeIndex:
        price = df['close']
        mean = price.rolling(lookback).mean()
        std = price.rolling(lookback).std()

        zscore = (price - mean) / std
        # Extreme deviation in either direction
        return df.index[np.abs(zscore) > z]

class LiquidityShockEvents(BaseEventGenerator):
    """
    Detects volume spikes, often indicating information arrival.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        volume = df['volume']
        # Use expanding window for normalization
        vol_mean = volume.expanding(min_periods=lookback).mean()
        vol_std = volume.expanding(min_periods=lookback).std()

        vol_std = vol_std.replace(0, np.nan)
        zscore = (volume - vol_mean) / vol_std
        return df.index[zscore > z]

class TimeEvents(BaseEventGenerator):
    """
    Control group: Clock-based sampling.
    """
    def generate(self, df: pd.DataFrame, step: int = 50) -> pd.DatetimeIndex:
        return df.index[::step]

class CusumEvents(BaseEventGenerator):
    """
    Wrapper for the existing CUSUM filter logic.
    """
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        # Defaults matching Layer 2 legacy
        k = params.get('k', 0.12) # Default threshold
        # We invoke generate_primary_signals
        # Note: generate_primary_signals expects 'volatility_1d' in df

        try:
            signals = generate_primary_signals(df, k=k)
            # signals is a DataFrame with 'consensus'
            if 'consensus' in signals.columns:
                # Events are where consensus != 0
                return signals.index[signals['consensus'] != 0]
        except Exception as e:
            print(f"CUSUM generation failed: {e}")

        return pd.DatetimeIndex([])

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
        self.family = family
        self.labeler_name = labeler_name
        self.params = params or {}

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
    # Assuming binary labels for this calculation
    mi = mutual_info_score(y1.loc[common], y2.loc[common])

    # Normalize by entropy of y1 to get range [0, 1] relative to the candidate
    entropy = shannon_entropy(y1.loc[common].value_counts())

    return mi / entropy if entropy > 0 else 0.0

def label_distribution_stable(labels: pd.Series, splits: int = 5, eps: float = 0.1) -> bool:
    """
    Checks if label distribution is stationary across time chunks.
    """
    if len(labels) < splits * 20: # Require decent sample size
        return True # Not enough data to fail check

    chunks = np.array_split(labels, splits)

    for a, b in combinations(chunks, 2):
        if len(a) < 10 or len(b) < 10:
            continue

        pa = a.value_counts(normalize=True)
        pb = b.value_counts(normalize=True)

        # Align indexes (ensure both have -1, 0, 1)
        pa, pb = pa.align(pb, fill_value=0)

        d = shannon_entropy(pa, pb)
        if not np.isfinite(d): # Handle distinct support
             d = 1.0

        if d > eps:
            return False
    return True


# ==========================================
# 3. Main Orchestration
# ==========================================

def orthogonal_label_generation(
    df: pd.DataFrame,
    labelers: Dict[str, Callable[[pd.DataFrame, pd.DatetimeIndex], pd.Series]],
    tau_uniqueness: float = 0.1,
    tau_mi: float = 0.1
) -> List[Geometry]:

    index = df.index

    # 1. Instantiate Generators
    event_families = {
        "CUSUM": CusumEvents(),
        "VOL": VolatilityShockEvents(),
        "TREND": TrendInitiationEvents(),
        "MEAN_REV": MeanReversionExtremeEvents(),
        "LIQUIDITY": LiquidityShockEvents(),
        "TIME": TimeEvents()
    }

    candidates = []

    # 2. Generate Candidates (Cartesian Product of Family x Labeler)
    print(f"Generating candidates from {len(event_families)} families x {len(labelers)} labelers...")

    for fam_name, fam in event_families.items():
        try:
            # Generate Events
            # Some generators might take specific params, using defaults here
            events = fam.generate(df)

            if len(events) < 10:
                print(f"Skipping family {fam_name}: Too few events ({len(events)})")
                continue

            for lbl_name, lbl_func in labelers.items():
                # Extract baked-in params if partial
                lbl_params = {}
                if isinstance(lbl_func, partial):
                    lbl_params = lbl_func.keywords

                # Generate Labels
                try:
                    # Expecting label_func(df, events) -> pd.Series
                    labels = lbl_func(df, events)
                except Exception as e:
                    print(f"Skipping {fam_name}_{lbl_name}: Label generation failed ({e})")
                    continue

                if labels.empty or labels.dropna().empty:
                    continue

                g = Geometry(
                    name=f"{fam_name}_{lbl_name}",
                    events=events,
                    labels=labels,
                    family=fam_name,
                    labeler_name=lbl_name,
                    params=lbl_params
                )
                g.indicator = build_indicator_matrix(events, index)
                candidates.append(g)

        except Exception as e:
            print(f"Error processing family {fam_name}: {e}")
            continue

    # 3. Filter for Uniqueness and Orthogonality
    accepted = []
    global_indicator = pd.DataFrame(index=index) # Tracks all accepted events so far

    print(f"Generated {len(candidates)} candidate geometries. Starting filter...")

    for g in candidates:
        # A. Check Marginal Uniqueness (vs everything already accepted)
        # If nothing accepted yet, uniqueness is 1.0 (self)
        if global_indicator.empty:
            uniq = 1.0
        else:
            temp_indicator = pd.concat([global_indicator, g.indicator], axis=1).fillna(0)
            uniq = average_uniqueness(temp_indicator)

        g.avg_uniqueness = uniq

        if uniq < tau_uniqueness:
            print(f"Rejected {g.name}: Uniqueness {uniq:.2f} < {tau_uniqueness}")
            continue

        # B. Check Mutual Information (Redundancy in outcome)
        redundant = False
        for a in accepted:
            mi = normalized_mi(g.labels, a.labels)
            if mi > tau_mi:
                print(f"Rejected {g.name}: High MI with {a.name} ({mi:.2f})")
                redundant = True
                break
        if redundant:
            continue

        # C. Check Stability
        if not label_distribution_stable(g.labels):
            print(f"Rejected {g.name}: Unstable label distribution")
            continue

        # Accept
        accepted.append(g)
        global_indicator[g.name] = g.indicator
        print(f"Accepted {g.name}")

    return accepted
