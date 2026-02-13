import bisect
from math import erf

import numpy as np
import pandas as pd


def _normal_cdf(x: pd.Series) -> pd.Series:
    arr = x.to_numpy(dtype=np.float64) / np.sqrt(2.0)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(arr))
    return pd.Series(cdf, index=x.index, dtype=np.float32)


def _rolling_percentile_exact(s: pd.Series, n: int) -> pd.Series:
    vals = s.to_numpy(dtype=np.float64)
    out = np.full(len(vals), 0.5, dtype=np.float32)
    window = []
    for i, cur in enumerate(vals):
        if i > 0:
            prev = vals[i - 1]
            bisect.insort(window, prev)
            if len(window) > n:
                old = vals[i - 1 - n]
                j = bisect.bisect_left(window, old)
                if j < len(window):
                    window.pop(j)
        if len(window) < n or not np.isfinite(cur):
            continue
        rank = bisect.bisect_right(window, cur)
        out[i] = rank / float(n)
    return pd.Series(out, index=s.index, dtype=np.float32)


def add_gate_features(
    df: pd.DataFrame,
    s_col: str,
    prefix: str,
    n: int = 256,
    add_strict: bool = True,
    percentile_mode: str = "approx",
    min_std: float = 1e-6,
) -> pd.DataFrame:
    if s_col not in df.columns:
        raise KeyError(f"Missing score column: {s_col}")

    s = df[s_col].astype(np.float32)

    roll = s.rolling(n, min_periods=n)
    mean = roll.mean().shift(1)
    std = roll.std(ddof=0).shift(1).clip(lower=min_std)
    z = ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    if percentile_mode == "exact":
        pct = _rolling_percentile_exact(s, n)
    else:
        pct = _normal_cdf(z).clip(0.0, 1.0).fillna(0.5).astype(np.float32)

    bins = np.digitize(pct.to_numpy(), bins=[1.0 / 3.0, 2.0 / 3.0]).astype(np.int8)

    df[f"{prefix}_mean_{n}"] = mean.fillna(0.0).astype(np.float32)
    df[f"{prefix}_std_{n}"] = std.fillna(min_std).astype(np.float32)
    df[f"{prefix}_z_{n}"] = z
    df[f"{prefix}_pct_{n}"] = pct.astype(np.float32)
    df[f"{prefix}_bin3_{n}"] = bins

    if add_strict:
        df[f"{prefix}_gt66_{n}"] = (pct > 0.66).astype(np.int8)
        df[f"{prefix}_gt85_{n}"] = (pct > 0.85).astype(np.int8)

    return df


def add_accept_gate_features(
    df: pd.DataFrame,
    s_col: str = "accept_score",
    N: int = 256,
    add_strict: bool = True,
    percentile_mode: str = "approx",
) -> pd.DataFrame:
    return add_gate_features(
        df=df,
        s_col=s_col,
        prefix="s",
        n=N,
        add_strict=add_strict,
        percentile_mode=percentile_mode,
    )


def cross_sectional_gate_aggregates(x: pd.DataFrame, trim_q: float = 0.10) -> pd.DataFrame:
    """Robust cross-sectional aggregates per timestamp for gate drivers."""
    if x.empty:
        return pd.DataFrame(index=x.index)

    lo = x.quantile(trim_q, axis=1)
    hi = x.quantile(1.0 - trim_q, axis=1)
    trimmed = x.where(x.ge(lo, axis=0) & x.le(hi, axis=0))

    out = pd.DataFrame(index=x.index)
    out["cs_median"] = x.median(axis=1)
    out["cs_trimmed_mean"] = trimmed.mean(axis=1)
    out["cs_p75"] = x.quantile(0.75, axis=1)
    out["cs_p90"] = x.quantile(0.90, axis=1)
    out["cs_iqr"] = (x.quantile(0.75, axis=1) - x.quantile(0.25, axis=1))
    out["cs_std"] = x.std(axis=1)
    out = out.fillna(0.0).astype(np.float32)
    return out


def gate_stability_diagnostics(s: pd.Series, bin_col: pd.Series | None = None) -> dict:
    """Simple diagnostics for gate stability and bin occupancy."""
    s = s.astype(np.float32)
    out = {
        "lag1_autocorr": float(s.autocorr(lag=1)) if len(s) > 2 else 0.0,
        "mean": float(s.mean()),
        "std": float(s.std()),
    }
    monthly = s.resample("MS").agg(["mean", "std"])
    out["monthly_mean_std"] = float(monthly["mean"].std()) if not monthly.empty else 0.0
    out["monthly_std_mean"] = float(monthly["std"].mean()) if not monthly.empty else 0.0
    if bin_col is not None:
        vc = pd.Series(bin_col).value_counts(normalize=True)
        out["bin3_share_0"] = float(vc.get(0, 0.0))
        out["bin3_share_1"] = float(vc.get(1, 0.0))
        out["bin3_share_2"] = float(vc.get(2, 0.0))
    return out


def conditional_uplift_by_bin(pred: pd.Series, ret: pd.Series, bins: pd.Series) -> pd.DataFrame:
    """Per-bin uplift diagnostics for walk-forward analysis."""
    df = pd.DataFrame({"pred": pred, "ret": ret, "bin": bins}).dropna()
    rows = []
    for b, g in df.groupby("bin"):
        if len(g) < 2:
            continue
        ic = g["pred"].corr(g["ret"], method="spearman")
        p10 = g.nlargest(max(1, int(0.1 * len(g))), "pred")["ret"].mean()
        rows.append({"bin": int(b), "n": int(len(g)), "ic": float(ic if np.isfinite(ic) else 0.0), "prec10_ret": float(p10)})
    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
