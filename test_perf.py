import numpy as np
import pandas as pd
from extreme_price_movements.offline_optimisers.preds_metrics_computations import _safe_corr
import time

def test_perf():
    np.random.seed(42)
    n_rows = 1000000
    n_groups = 10000

    df = pd.DataFrame({
        "ts": np.random.randint(0, n_groups, n_rows),
        "score": np.random.randn(n_rows),
        "fwd_ret": np.random.randn(n_rows)
    })

    g = df.groupby("ts", sort=False)

    start = time.time()
    ic_series = g.apply(lambda x: _safe_corr(x["score"].to_numpy(), x["fwd_ret"].to_numpy()))
    end = time.time()
    print(f"Pandas apply: {end - start:.4f}s")

    start = time.time()
    # Fast approach:
    # We can use np.bincount or similar, or Numba.
    # Since we need correlation per group...

    def fast_corr(df, by, ret_col):
        score = df["score"].to_numpy()
        ret = df[ret_col].to_numpy()
        by_col = df[by].to_numpy()

        # factorize
        codes, _ = pd.factorize(by_col, sort=False)
        n_groups = codes.max() + 1

        # bincounts
        n = np.bincount(codes, minlength=n_groups)
        sum_x = np.bincount(codes, weights=score, minlength=n_groups)
        sum_y = np.bincount(codes, weights=ret, minlength=n_groups)
        sum_xx = np.bincount(codes, weights=score*score, minlength=n_groups)
        sum_yy = np.bincount(codes, weights=ret*ret, minlength=n_groups)
        sum_xy = np.bincount(codes, weights=score*ret, minlength=n_groups)

        # Avoid div by 0 for n < 3
        valid = n >= 3
        n_v = n[valid]

        mean_x = sum_x[valid] / n_v
        mean_y = sum_y[valid] / n_v

        cov = (sum_xy[valid] / n_v) - mean_x * mean_y
        var_x = (sum_xx[valid] / n_v) - mean_x * mean_x
        var_y = (sum_yy[valid] / n_v) - mean_y * mean_y

        # stds
        std_x = np.sqrt(np.maximum(var_x, 0))
        std_y = np.sqrt(np.maximum(var_y, 0))

        denom = std_x * std_y

        # denom > 0 mask
        valid_denom = denom > 1e-12
        corr = np.full(len(n), np.nan)

        # calculate
        valid_idx = np.where(valid)[0][valid_denom]
        corr[valid_idx] = cov[valid_denom] / denom[valid_denom]

        return corr

    res = fast_corr(df, "ts", "fwd_ret")
    end = time.time()
    print(f"Fast vectorized: {end - start:.4f}s")

test_perf()
