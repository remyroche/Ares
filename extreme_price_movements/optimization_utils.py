import heapq
import numpy as np
import pandas as pd
from extreme_price_movements.utils import tprint

def filter_low_variance_assets(
    store,
    syms,
    lookback_days=30,
    threshold_pct=0.40,
    ts_sig=None,
    sample_stride=1,
):
    """
    Loads 'close' for all syms, resamples to 12H, computes variance.
    Returns top threshold_pct symbols by variance.
    """
    stride = max(1, int(sample_stride))
    reason_counts = {
        "load_error": 0,
        "too_few_rows": 0,
        "too_few_sampled_points": 0,
        "near_zero_variance": 0,
        "accepted": 0,
    }
    tprint(
        f"Filtering {len(syms)} symbols by variance (Top {int(threshold_pct*100)}%, stride={stride})..."
    )
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
                reason_counts["too_few_rows"] += 1
                continue

            close_series = df["close"]
            if stride > 1:
                # Coarse universe gating only needs a sparse sample of the recent path.
                r = close_series.iloc[::stride].to_numpy()
            else:
                # keep pandas resample for correct 12H anchoring
                r = close_series.resample("12h").last().to_numpy()

            if r.size < 3:
                reason_counts["too_few_sampled_points"] += 1
                continue

            rets = r[1:] / r[:-1] - 1.0
            var = float(np.var(rets, ddof=1)) if rets.size > 1 else 0.0
            
            # Filter strictly constant assets (or near constant)
            if var > 1e-18:
                variances.append((var, s))
                reason_counts["accepted"] += 1
            else:
                reason_counts["near_zero_variance"] += 1

        except Exception as e:
            # tprint(f"Error checking variance {s}: {e}")
            reason_counts["load_error"] += 1
            pass

    if not variances:
        tprint(
            "Variance Filter Reasons: "
            + ", ".join(f"{k}={v}" for k, v in reason_counts.items())
        )
        return syms # Fallback

    # Optimization: Use heapq for top K
    # Calculate n_keep based on valid variances to avoid keeping too few/many if failures occur
    n_keep = max(1, int(len(variances) * threshold_pct))
    top = heapq.nlargest(n_keep, variances, key=lambda x: x[0])
    top_syms = [x[1] for x in top]

    tprint(f"Variance Filter: Kept {len(top_syms)}/{len(syms)} symbols.")
    tprint(
        "Variance Filter Reasons: "
        + ", ".join(f"{k}={v}" for k, v in reason_counts.items())
    )
    return sorted(top_syms)
