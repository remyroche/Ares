import pandas as pd
from extreme_price_movements.utils import tprint

def filter_low_variance_assets(store, syms, lookback_days=30, threshold_pct=0.40):
    """
    Loads 'close' for all syms, resamples to 12H, computes variance.
    Returns top threshold_pct symbols by variance.
    """
    tprint(f"Filtering {len(syms)} symbols by variance (Top {int(threshold_pct*100)}%)...")
    variances = []

    # We need a reference date to limit load?
    # Partitioned store load reads all unless we check files.
    # But load returns full df.
    # Reading 'close' only is faster IO.

    # We can't efficiently load "last 30 days" without scanning files.
    # Assuming standard load is fast enough for metadata or we rely on full load.
    # With `load(columns=['close'])` it should be faster.

    for s in syms:
        try:
            # We assume load handles caching or is fast enough for 300 symbols.
            # If partitions are year/month, it reads all months.
            # This might be slow if history is huge.
            # Optimization: If we had a way to load only recent partitions.
            # The current `load` implementation reads ALL partitions.
            # For 4 years of 1h data (35k rows), it's manageable.

            df = store.load(s, columns=["close"])
            if df.empty:
                continue

            # Filter last lookback_days
            cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
            subset = df.loc[cutoff:]

            if len(subset) < 10:
                continue

            # Resample 12H
            resampled = subset["close"].resample("12h").last()

            # Variance of returns
            var = resampled.pct_change().var()
            variances.append((var, s))

        except Exception as e:
            # tprint(f"Error checking variance {s}: {e}")
            pass

    if not variances:
        return syms # Fallback

    variances.sort(key=lambda x: x[0], reverse=True)

    n_keep = int(len(syms) * threshold_pct)
    top_syms = [x[1] for x in variances[:n_keep]]

    tprint(f"Variance Filter: Kept {len(top_syms)}/{len(syms)} symbols.")
    return sorted(top_syms)
