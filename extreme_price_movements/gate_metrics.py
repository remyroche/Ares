import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from scipy.stats import spearmanr, median_abs_deviation

def compute_stage_gate_metrics(y_true, y_prob, y_ret=None, model_type="classifier", baseline_prev=None, cv_prec10=None):
    """
    Compute stage gate metrics for Alpha (classifier) or Meta (quantile) models.
    Returns a dict with metrics and pass/fail flags.
    """
    metrics = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Common validity check
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if y_ret is not None:
        y_ret = np.asarray(y_ret)
        mask &= np.isfinite(y_ret)

    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_ret is not None:
        y_ret = y_ret[mask]

    if len(y_true) < 10:
        return {"passed": False, "reason": "Insufficient data", "metrics": metrics}

    if model_type == "classifier":
        # Binarize soft labels (continuous 0-1 from soft label blending)
        # sklearn's brier_score_loss / log_loss require binary y_true
        y_bin = (y_true >= 0.5).astype(np.float64)

        # Baseline prevalence
        base_prev = np.mean(y_bin) if baseline_prev is None else baseline_prev
        metrics["Base_Prev"] = base_prev

        # 1. PR-AUC
        try:
            pr_auc = average_precision_score(y_bin, y_prob)
        except:
            pr_auc = 0.0

        pr_auc_thresh = max(1.25 * base_prev, base_prev + 0.05)
        metrics["PR_AUC"] = pr_auc
        metrics["PR_AUC_Thresh"] = pr_auc_thresh
        pass_pr_auc = pr_auc >= pr_auc_thresh
        metrics["Pass_PR_AUC"] = pass_pr_auc

        # 2. Brier & LogLoss Improvement
        brier = brier_score_loss(y_bin, y_prob)
        ll = log_loss(y_bin, np.clip(y_prob, 1e-7, 1-1e-7))

        base_prob = np.full_like(y_prob, base_prev)
        base_brier = brier_score_loss(y_bin, base_prob)
        base_ll = log_loss(y_bin, base_prob)

        brier_imp = (base_brier - brier) / base_brier if base_brier > 1e-9 else 0.0
        ll_imp = (base_ll - ll) / base_ll if base_ll > 1e-9 else 0.0

        metrics["Brier"] = brier
        metrics["Brier_Imp"] = brier_imp
        metrics["LogLoss"] = ll
        metrics["LogLoss_Imp"] = ll_imp
        pass_loss = (brier_imp >= 0.02) and (ll_imp >= 0.02)
        metrics["Pass_Loss"] = pass_loss

        # 3. Lift@k and Precision@k Lift
        k = 0.20
        n_k = max(1, int(len(y_bin) * k))
        idx = np.argsort(y_prob)[-n_k:]
        prec_k = np.mean(y_bin[idx])

        lift_k = prec_k / base_prev if base_prev > 1e-9 else 0.0
        prec_lift_abs = prec_k - base_prev

        # User: Lift@k >= 1.2 AND Precision@k lift >= +5% (or +2–3pp absolute)
        # Interpretation: Lift@k >= 1.2 (mandatory) AND (Prec_Lift_Rel >= 0.05 OR Prec_Lift_Abs >= 0.02)
        # Note: Lift@k >= 1.2 implies Prec_Lift_Rel >= 0.20, so the second condition is implicitly met unless
        # "Precision@k lift >= +5%" means something else. Assuming standard interpretation, it's redundant but we implement as requested.

        metrics["Lift_k"] = lift_k
        metrics["Prec_k"] = prec_k
        metrics["Prec_k_Lift_Abs"] = prec_lift_abs

        pass_lift_primary = lift_k >= 1.2
        pass_lift_secondary = (lift_k >= 1.05) or (prec_lift_abs >= 0.02)

        pass_lift = pass_lift_primary and pass_lift_secondary
        metrics["Pass_Lift"] = pass_lift

        # 4. CV(Precision@k)
        metrics["CV_Prec_k"] = cv_prec10 if cv_prec10 is not None else 1.0
        pass_cv = (metrics["CV_Prec_k"] <= 0.30)
        metrics["Pass_CV"] = pass_cv

        metrics["passed"] = bool(pass_pr_auc and pass_loss and pass_lift and pass_cv)

    elif model_type == "quantile_meta":
        # Target is y_true (rank/score). y_prob is prediction.
        tau = 0.85

        # 1. Coverage
        cov = np.mean(y_true <= y_prob)
        metrics["Coverage"] = cov
        metrics["Coverage_Diff"] = abs(cov - tau)
        pass_cov = metrics["Coverage_Diff"] <= 0.05
        metrics["Pass_Coverage"] = pass_cov

        # 2. Pinball Improvement
        def pinball(y, q, alpha):
            return np.mean(np.maximum(alpha * (y - q), (alpha - 1.0) * (y - q)))

        pb = pinball(y_true, y_prob, tau)
        # Baseline: constant prediction at quantile(tau)
        base_pred = np.quantile(y_true, tau)
        base_pb = pinball(y_true, np.full_like(y_true, base_pred), tau)

        pb_imp = (base_pb - pb) / base_pb if base_pb > 1e-9 else 0.0
        metrics["Pinball_Imp"] = pb_imp
        pass_pb = pb_imp >= 0.02 # or >= 2/3 folds? We check global improvement here.
        metrics["Pass_Pinball"] = pass_pb

        # 3. Spearman IC
        ic, _ = spearmanr(y_true, y_prob)
        metrics["Spearman_IC"] = ic
        pass_ic = (ic >= 0.04)
        metrics["Pass_IC"] = pass_ic

        # 4. Top20 - Bot50 Median Spread
        n_20 = max(1, int(len(y_true) * 0.20))
        n_50 = max(1, int(len(y_true) * 0.50))

        idx_top = np.argsort(y_prob)[-n_20:]
        idx_bot = np.argsort(y_prob)[:n_50] # bottom 50% by prediction

        med_top = np.median(y_true[idx_top])
        med_bot = np.median(y_true[idx_bot])
        spread = med_top - med_bot

        mad_y = median_abs_deviation(y_true, scale='normal')
        thresh = max(0.0, 0.25 * mad_y)

        metrics["Spread"] = spread
        metrics["Spread_Thresh"] = thresh
        pass_spread = (spread >= thresh) or (spread > 0) # "or at least > 0" from user request
        metrics["Pass_Spread"] = pass_spread

        # 5. Conditional Downside (ES10 on Top 20% Selection)
        # Using y_ret if available (raw returns), else y_true
        target_y = y_ret if y_ret is not None else y_true

        # ES10 of baseline (unconditional)
        q10_base = np.quantile(target_y, 0.10)
        # Expected Shortfall (CVaR) is average of returns <= VaR
        # For returns, lower is worse.
        mask_base = target_y <= q10_base
        es10_base = np.mean(target_y[mask_base]) if mask_base.any() else q10_base

        # ES10 of Top 20% selected by model
        idx_sel = np.argsort(y_prob)[-n_20:]
        sel_y = target_y[idx_sel]

        if len(sel_y) < 5:
            es10_sel = es10_base
        else:
            q10_sel = np.quantile(sel_y, 0.10)
            mask_sel = sel_y <= q10_sel
            es10_sel = np.mean(sel_y[mask_sel]) if mask_sel.any() else q10_sel

        metrics["ES10_Base"] = es10_base
        metrics["ES10_Sel"] = es10_sel

        # Logic: es10_sel should be not worse than es10_base by > 20%
        # Case 1: ES is negative (loss). e.g. -0.10. Worse is -0.12.
        # Limit = -0.10 - 0.20 * |-0.10| = -0.12.
        # Check: es10_sel >= -0.12.
        # Case 2: ES is positive (profit). e.g. 0.05. Worse is 0.04.
        # Limit = 0.05 - 0.20 * |0.05| = 0.04.
        # Check: es10_sel >= 0.04.

        limit = es10_base - 0.20 * abs(es10_base)
        pass_downside = es10_sel >= limit
        metrics["Pass_Downside"] = pass_downside

        metrics["passed"] = bool(pass_cov and pass_pb and pass_ic and pass_spread and pass_downside)

    return metrics
