import heapq
import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint

def filter_low_variance_assets(store, syms, lookback_days=30, threshold_pct=0.40, ts_sig=None):
    """
    Loads 'close' for all syms, resamples to 12H, computes variance.
    Returns top threshold_pct symbols by variance.
    """
    tprint(f"Filtering {len(syms)} symbols by variance (Top {int(threshold_pct*100)}%)...")
    variances = []

    if ts_sig is None:
        asof = pd.Timestamp.utcnow()
    else:
        asof = ts_sig

    cutoff = asof - pd.Timedelta(days=lookback_days)

    for s in syms:
        try:
            # Optimization: Load only relevant time range
            # Uses partition pruning in store
            df = store.load(s, columns=["close"], start_ts=cutoff, end_ts=asof)
            if df.empty or len(df) < 10:
                continue

            # Optimization: Resample 12H and use numpy for variance
            # keep pandas resample for correct 12H anchoring
            r = df["close"].resample("12h").last().to_numpy()

            if r.size < 3:
                continue

            rets = r[1:] / r[:-1] - 1.0
            var = float(np.var(rets, ddof=1)) if rets.size > 1 else 0.0
            
            # Filter strictly constant assets (or near constant)
            if var > 1e-18:
                variances.append((var, s))

        except Exception as e:
            # tprint(f"Error checking variance {s}: {e}")
            pass

    if not variances:
        return syms # Fallback

    # Optimization: Use heapq for top K
    # Calculate n_keep based on valid variances to avoid keeping too few/many if failures occur
    n_keep = max(1, int(len(variances) * threshold_pct))
    top = heapq.nlargest(n_keep, variances, key=lambda x: x[0])
    top_syms = [x[1] for x in top]

    tprint(f"Variance Filter: Kept {len(top_syms)}/{len(syms)} symbols.")
    return sorted(top_syms)
