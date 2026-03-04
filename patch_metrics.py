import re

with open('extreme_price_movements/position_sizer/models.py', 'r') as f:
    content = f.read()

# Add log_loss, roc_auc_score, average_precision_score to imports if not there
if 'average_precision_score' not in content:
    content = content.replace('from sklearn.metrics import roc_auc_score', 'from sklearn.metrics import roc_auc_score, log_loss, average_precision_score')

# Update _compute_metrics
search = """def _compute_metrics(prefix, y_true, y_prob, sample_weight=None, y_hard_ref=None):
    tprint(f"Entering function: _compute_metrics in {__name__}")
    tprint(f"_compute_metrics params: prefix={prefix}, y_true={y_true}, y_prob={y_prob}, sample_weight={sample_weight}, y_hard_ref={y_hard_ref}")
    out = {
        f"bce_{prefix}": _bce_soft(y_prob, y_true, sample_weight=sample_weight),
        f"brier_{prefix}": _weighted_mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2, sample_weight),
        f"ece_{prefix}": _ece(y_true, y_prob, n_bins=20, sample_weight=sample_weight),
        f"spearman_{prefix}": _safe_corr(y_prob, y_true),
    }
    if y_hard_ref is not None and len(y_hard_ref) == len(y_prob) and len(np.unique(y_hard_ref)) > 1:
        try:
            out[f"auc_{prefix}"] = float(roc_auc_score(np.asarray(y_hard_ref, dtype=int), y_prob))
        except Exception:
            out[f"auc_{prefix}"] = float("nan")
    return out"""

replace = """def _compute_metrics(prefix, y_true, y_prob, sample_weight=None, y_hard_ref=None):
    tprint(f"Entering function: _compute_metrics in {__name__}")
    tprint(f"_compute_metrics params: prefix={prefix}, y_true={y_true}, y_prob={y_prob}, sample_weight={sample_weight}, y_hard_ref={y_hard_ref}")
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    out = {
        f"logloss_{prefix}": _bce_soft(y_prob_arr, y_true_arr, sample_weight=sample_weight),
        f"brier_{prefix}": _weighted_mean((y_true_arr - y_prob_arr) ** 2, sample_weight),
        f"ece_{prefix}": _ece(y_true_arr, y_prob_arr, n_bins=20, sample_weight=sample_weight),
        f"spearman_{prefix}": _safe_corr(y_prob_arr, y_true_arr),
    }
    # Backward compat
    out[f"bce_{prefix}"] = out[f"logloss_{prefix}"]

    if y_hard_ref is not None and len(y_hard_ref) == len(y_prob) and len(np.unique(y_hard_ref)) > 1:
        try:
            out[f"roc_auc_{prefix}"] = float(roc_auc_score(np.asarray(y_hard_ref, dtype=int), y_prob_arr))
        except Exception:
            out[f"roc_auc_{prefix}"] = float("nan")
        try:
            out[f"pr_auc_{prefix}"] = float(average_precision_score(np.asarray(y_hard_ref, dtype=int), y_prob_arr))
        except Exception:
            out[f"pr_auc_{prefix}"] = float("nan")
        # Backward compat
        out[f"auc_{prefix}"] = out[f"roc_auc_{prefix}"]
    return out"""

content = content.replace(search, replace)

with open('extreme_price_movements/position_sizer/models.py', 'w') as f:
    f.write(content)
