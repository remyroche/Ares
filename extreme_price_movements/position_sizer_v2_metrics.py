import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

def compute_top_slice_metrics(scores: np.ndarray, realized_returns: np.ndarray, top_fracs: Tuple[float, ...] = (0.1, 0.2)) -> Dict[str, float]:
    """Computes mean realized returns and hit rates for top quantiles."""
    metrics = {}
    n = len(scores)
    if n == 0:
        for f in top_fracs:
            metrics[f"top_{int(f*100)}_mean_net"] = 0.0
            metrics[f"top_{int(f*100)}_hit_rate"] = 0.0
        return metrics

    for f in top_fracs:
        k = max(1, int(n * f))
        idx = np.argpartition(scores, -k)[-k:]
        rets = realized_returns[idx]
        metrics[f"top_{int(f*100)}_mean_net"] = float(np.mean(rets))
        metrics[f"top_{int(f*100)}_hit_rate"] = float(np.mean(rets > 0))
    return metrics

def compute_bucket_monotonicity(scores: np.ndarray, realized_values: np.ndarray, n_buckets: int = 10) -> float:
    """
    Groups predictions into quantiles and computes rank correlation
    between bucket rank and mean realized value.
    """
    if len(scores) < n_buckets:
        return 0.0
    try:
        labels = pd.qcut(scores, q=n_buckets, labels=False, duplicates='drop')
        df = pd.DataFrame({"score": scores, "realized": realized_values, "bucket": labels})
        bucket_means = df.groupby("bucket")["realized"].mean().values
        spearman, _ = pd.Series(bucket_means).corr(pd.Series(np.arange(len(bucket_means))), method="spearman")
        return float(spearman) if pd.notna(spearman) else 0.0
    except Exception:
        return 0.0

def compute_false_safe_rate(pred_downside: np.ndarray, realized_downside: np.ndarray, low_q: float = 0.2, high_q: float = 0.8) -> float:
    """
    Computes fraction of predictions that indicated low downside risk (bottom `low_q`)
    but actually realized severe downside risk (top `high_q`).
    Downside metrics: higher value = more downside risk (e.g. MAE).
    """
    if len(pred_downside) < 10:
        return 0.0

    safe_thresh = np.percentile(pred_downside, low_q * 100)
    danger_thresh = np.percentile(realized_downside, high_q * 100)

    predicted_safe = pred_downside <= safe_thresh
    n_safe = np.sum(predicted_safe)

    if n_safe == 0:
        return 0.0

    actually_dangerous = realized_downside >= danger_thresh
    false_safe = predicted_safe & actually_dangerous

    return float(np.sum(false_safe) / n_safe)

def compute_uncertainty_calibration(pred_unc: np.ndarray, realized_abs_err: np.ndarray, n_buckets: int = 10) -> Dict[str, float]:
    """
    Computes calibration metrics for uncertainty predictions vs realized absolute errors.
    """
    metrics = {}
    if len(pred_unc) < n_buckets:
        return metrics

    try:
        corr = pd.Series(pred_unc).corr(pd.Series(realized_abs_err), method="spearman")
        metrics["spearman_corr"] = float(corr) if pd.notna(corr) else 0.0

        # Underestimation penalty rate: fraction where realized error is significantly > predicted uncertainty
        # Assuming both are roughly normalized/on same scale. If log1p, we expm1 them first.
        # For a generic metric, we say underestimated if realized > 1.5 * predicted
        underest = realized_abs_err > (1.5 * pred_unc)
        metrics["underestimation_rate"] = float(np.mean(underest))

        # Top decile
        k = max(1, int(len(pred_unc) * 0.10))
        idx = np.argpartition(pred_unc, -k)[-k:]
        metrics["top_decile_mean_realized_err"] = float(np.mean(realized_abs_err[idx]))
        metrics["full_sample_mean_realized_err"] = float(np.mean(realized_abs_err))

    except Exception:
        pass

    return metrics
