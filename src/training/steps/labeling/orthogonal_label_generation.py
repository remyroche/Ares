import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy as shannon_entropy
from typing import List, Dict, Union, Callable, Any
from functools import partial
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import CUSUM generator dependency
try:
    from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
except ImportError:
    generate_primary_signals = None

# ==========================================
# 1. Event Generators (Orthogonal Families)
# ==========================================

class BaseEventGenerator:
    """
    Abstract base class for event generation strategies.
    """
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        raise NotImplementedError

class SymmetricCusumEvents(BaseEventGenerator):
    """
    The De Prado Standard (Chapter 2). Detects structural breaks in the mean price.
    More robust to noise than Simple Moving Average crossovers.
    """
    def generate(self, df: pd.DataFrame, h: float = 0.05) -> pd.DatetimeIndex:
        # h is the threshold in percent (e.g., 0.05 = 5% deviation triggers event)
        # In practice, we often set h based on daily volatility (e.g., h = vol * 2)
        price = df['close']
        t_events = []
        s_pos = 0
        s_neg = 0

        # diff = price.pct_change().diff() # or raw log-returns
        diff = price.pct_change() # using simple returns for this implementation

        # Calculate dynamic threshold based on rolling vol (optional but recommended)
        # Here we use fixed 'h' for simplicity, or you can pass a vol series

        # Using loop as per instructions, but accessing values for speed
        diff_vals = diff.values
        index = diff.index

        # Skip first element (NaN)
        for i in range(1, len(diff)):
            r = diff_vals[i]
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            if s_pos > h:
                s_neg = 0
                s_pos = 0
                t_events.append(index[i])
            elif s_neg < -h:
                s_neg = 0
                s_pos = 0
                t_events.append(index[i])

        return pd.DatetimeIndex(t_events)

class ImprovedCUSUMEvents(BaseEventGenerator):
    """
    Wrapper for the existing CUSUM filter logic (Layer 2 pre-existing).
    """
    def generate(self, df: pd.DataFrame, **params) -> pd.DatetimeIndex:
        # Defaults matching Layer 2 legacy
        k = params.get('k', 0.12) # Default threshold

        if generate_primary_signals is None:
            logger.warning("generate_primary_signals not available, skipping ImprovedCUSUMEvents")
            return pd.DatetimeIndex([])

        try:
            signals = generate_primary_signals(df, k=k)
            # signals is a DataFrame with 'consensus'
            if 'consensus' in signals.columns:
                # Events are where consensus != 0
                return signals.index[signals['consensus'] != 0]
        except Exception as e:
            logger.warning(f"CUSUM generation failed: {e}")

        return pd.DatetimeIndex([])

class HurstStateEvents(BaseEventGenerator):
    """
    Detects when the market switches from "Random Walk" to "Trend".
    Triggers when Hurst Exponent crosses critical thresholds.
    """
    def get_hurst(self, series):
        # Simplified R/S analysis or similar
        # (Using a quick approximation for performance in loops)
        lags = range(2, 20)

        # Convert to numpy for performance
        arr = series.values

        # Calculate std of diffs at various lags
        # tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
        tau = []
        for lag in lags:
            if len(arr) > lag:
                diff = arr[lag:] - arr[:-lag]
                tau.append(np.sqrt(np.std(diff)))
            else:
                tau.append(np.nan)

        # Filter NaNs/Zeros/Infs for log
        lags_clean = []
        tau_clean = []
        for l, t in zip(lags, tau):
            if t > 0 and np.isfinite(t):
                lags_clean.append(l)
                tau_clean.append(t)

        if len(lags_clean) < 2:
            return 0.5

        try:
            poly = np.polyfit(np.log(lags_clean), np.log(tau_clean), 1)
            return poly[0] * 2.0
        except Exception:
            return 0.5

    def generate(self, df: pd.DataFrame, lookback: int = 100, threshold: float = 0.6) -> pd.DatetimeIndex:
        # Warning: Hurst is computationally expensive.
        # rolling_apply is slow. We generate events sparsely.
        price = df['close']

        # Use rolling apply
        hurst_vals = price.rolling(lookback).apply(self.get_hurst, raw=False)

        # Trigger when we cross INTO a trend regime (H > threshold)
        # We only want the initiation of the regime, not every day inside it.
        trigger = (hurst_vals > threshold) & (hurst_vals.shift(1) <= threshold)

        return price.index[trigger]

class VolatilityShockEvents(BaseEventGenerator):
    """
    Detects sudden spikes in volatility relative to the running history.
    """
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
    """
    Detects moving average crossovers indicating a regime shift.
    """
    def generate(self, df: pd.DataFrame, short: int = 20, long: int = 100) -> pd.DatetimeIndex:
        price = df['close']
        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()

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
        return df.index[np.abs(zscore) > z]

class LiquidityShockEvents(BaseEventGenerator):
    """
    Detects volume spikes.
    """
    def generate(self, df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> pd.DatetimeIndex:
        if 'volume' not in df.columns:
            return pd.DatetimeIndex([])
        volume = df['volume']
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
        self.score = 0.0

def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.Series:
    ind = pd.Series(0, index=index)
    valid_events = events.intersection(index)
    ind.loc[valid_events] = 1
    return ind

def average_uniqueness(indicators: pd.DataFrame) -> float:
    if indicators.empty:
        return 0.0
    concurrency = indicators.sum(axis=1)
    uniq = indicators.div(concurrency, axis=0).replace([np.inf, np.nan], 0)
    valid = indicators > 0
    if not valid.any().any():
        return 0.0
    return uniq[indicators > 0].mean().mean()

def normalized_mi(y1: pd.Series, y2: pd.Series) -> float:
    common = y1.index.intersection(y2.index)
    if len(common) < 10:
        return 0.0
    mi = mutual_info_score(y1.loc[common], y2.loc[common])
    entropy = shannon_entropy(y1.loc[common].value_counts())
    return mi / entropy if entropy > 0 else 0.0

# ==========================================
# 3. Main Orchestration
# ==========================================

def orthogonal_label_generation(
    df: pd.DataFrame,
    labelers: Dict[str, Callable[[pd.DataFrame, pd.DatetimeIndex], pd.Series]],
    scorer: Callable[[pd.DataFrame, pd.DatetimeIndex, pd.Series], float] = None,
    tau_uniqueness: float = 0.1,
    tau_mi: float = 0.5 # Relaxed from 0.1 to allow some correlation
) -> List[Geometry]:

    index = df.index

    # 1. Instantiate Generators with Variations (The Tournament Candidates)
    # "VOL_FAST, VOL_MED, VOL_SLOW" etc.

    generators = {
        # Volatility Shocks
        "VOL_FAST": partial(VolatilityShockEvents().generate, lookback=20, z=2.0),
        "VOL_MED": partial(VolatilityShockEvents().generate, lookback=50, z=2.0),
        "VOL_SLOW": partial(VolatilityShockEvents().generate, lookback=100, z=2.0),

        # Trend Initiation
        "TREND_FAST": partial(TrendInitiationEvents().generate, short=10, long=30),
        "TREND_MED": partial(TrendInitiationEvents().generate, short=20, long=60),
        "TREND_SLOW": partial(TrendInitiationEvents().generate, short=50, long=200),

        # Mean Reversion
        "MEAN_REV_FAST": partial(MeanReversionExtremeEvents().generate, lookback=20, z=2.0),
        "MEAN_REV_MED": partial(MeanReversionExtremeEvents().generate, lookback=50, z=2.5),
        "MEAN_REV_SLOW": partial(MeanReversionExtremeEvents().generate, lookback=100, z=3.0),

        # Liquidity
        "LIQ_FAST": partial(LiquidityShockEvents().generate, lookback=20, z=2.0),
        "LIQ_MED": partial(LiquidityShockEvents().generate, lookback=50, z=2.5),

        # New Generators
        "SYM_CUSUM_5": partial(SymmetricCusumEvents().generate, h=0.05),
        "SYM_CUSUM_2": partial(SymmetricCusumEvents().generate, h=0.02),

        "IMP_CUSUM": ImprovedCUSUMEvents().generate,

        "HURST_100": partial(HurstStateEvents().generate, lookback=100, threshold=0.6),
        "HURST_200": partial(HurstStateEvents().generate, lookback=200, threshold=0.6),

        # Control
        "TIME": TimeEvents().generate
    }

    candidates = []

    # 2. Generate Candidates (Cartesian Product of Family x Labeler)
    print(f"Generating candidates from {len(generators)} families x {len(labelers)} labelers...")

    for gen_name, gen_func in generators.items():
        try:
            # Generate Events
            try:
                if isinstance(gen_func, partial):
                    events = gen_func(df)
                else:
                    events = gen_func(df)
            except Exception as e:
                # Some generators might fail if missing columns (e.g. volume)
                continue

            if len(events) < 10:
                continue

            for lbl_name, lbl_func in labelers.items():
                # Extract baked-in params if partial
                lbl_params = {}
                if isinstance(lbl_func, partial):
                    lbl_params = lbl_func.keywords

                # Generate Labels
                try:
                    labels = lbl_func(df, events)
                except Exception as e:
                    print(f"Skipping {gen_name}_{lbl_name}: Label generation failed ({e})")
                    continue

                if labels.empty or labels.dropna().empty:
                    continue

                # Check for minimum positive labels
                if labels.sum() < 5:
                    continue

                g = Geometry(
                    name=f"{gen_name}_{lbl_name}",
                    events=events,
                    labels=labels,
                    family=gen_name.split('_')[0], # Group by broad family for reporting
                    labeler_name=lbl_name,
                    params=lbl_params
                )
                g.indicator = build_indicator_matrix(events, index)

                # Calculate Score (Learnability)
                if scorer:
                    try:
                        score = scorer(df, events, labels)
                        g.score = score
                    except Exception as e:
                        print(f"Scoring failed for {g.name}: {e}")
                        g.score = 0.0
                else:
                    g.score = 0.5

                candidates.append(g)

        except Exception as e:
            print(f"Error processing generator {gen_name}: {e}")
            continue

    # 3. Sort by Score
    candidates.sort(key=lambda x: x.score, reverse=True)

    # 4. Filter for Orthogonality
    accepted = []

    print(f"Generated {len(candidates)} candidate geometries. Starting filter (Tau_MI={tau_mi})...")

    # We maintain a list of accepted geometries to check MI against
    # The prompt says "The MI check detects this. It gets rejected."

    for g in candidates:
        if g.score < 0.51: # Minimum learnability threshold (random is 0.5)
             continue

        redundant = False
        for a in accepted:
            mi = normalized_mi(g.labels, a.labels)

            # Also check event overlap?
            # Prompt implies MI check on labels.
            # "It likely overlaps 80% with the winner. The MI check detects this."

            if mi > tau_mi:
                print(f"Rejected {g.name} (Score {g.score:.4f}): High MI with {a.name} ({mi:.2f})")
                redundant = True
                break

        if redundant:
            continue

        accepted.append(g)
        print(f"Accepted {g.name} (Score {g.score:.4f})")

    return accepted
