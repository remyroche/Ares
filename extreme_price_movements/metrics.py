import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import spearmanr
from .utils import tprint

class MetricsLogger:
    def __init__(self, log_dir="logs"):
        tprint(f"Entering function: __init__ in metrics.py")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self, ts: pd.Timestamp) -> str:
        tprint(f"Entering function: _get_log_path in metrics.py")
        date_str = ts.strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"metrics_{date_str}.csv")

    def log(self, ts_sig: pd.Timestamp, data: dict):
        """
        Logs metrics to a CSV file partitioned by day of ts_sig.
        data: dictionary of metrics
        """
        tprint(f"Entering function: log in metrics.py")
        # Ensure ts_sig is in data
        row = data.copy()
        row["ts_sig"] = ts_sig.isoformat()
        row["log_ts"] = datetime.now(timezone.utc).isoformat()

        path = self._get_log_path(ts_sig)
        df = pd.DataFrame([row])

        if not os.path.exists(path):
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)

        # Also tprint summary
        tprint(f"Metrics saved to {path}: {row}")


def _softsign(x):
    return x / (1.0 + np.abs(x))


def _zscore(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)


def _max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    if ec.size < 2:
        return 0.0
    peak = np.maximum.accumulate(ec)
    dd = peak - ec
    mdd = np.nanmax(dd)
    return float(mdd) if np.isfinite(mdd) else 0.0


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _normalize_bss(bss):
    bss_capped = float(np.clip(bss, -1.0, 1.0))
    return _clip01((bss_capped + 1.0) / 2.0)


def _position_from_prob(
    p,
    *,
    mode="linear",            # "linear" | "centered" | "rank" | "sigmoid"
    min_pos=0.0,
    max_pos=1.0,
    center=0.5,               # used by "centered"
    sigmoid_k=6.0,            # used by "sigmoid"
):
    """
    Map model probability/score to position size in [min_pos, max_pos] (or symmetric if min_pos<0).
    Assumes p is higher => stronger signal.
    """
    p = np.asarray(p, dtype=float)

    if mode == "linear":
        s = p
    elif mode == "centered":
        # -1..1 if p in [0,1], centered at `center`
        s = (p - center) / (0.5 if center == 0.5 else max(1e-12, max(center, 1 - center)))
        s = np.clip(s, -1.0, 1.0)
        # map -1..1 to min_pos..max_pos if symmetric bounds; else just scale into range
        if min_pos < 0.0 and max_pos > 0.0:
            # symmetric-ish: allow both long/short
            # s=-1 -> min_pos, s=+1 -> max_pos
            s = (s + 1.0) / 2.0
        else:
            # long-only
            s = (s + 1.0) / 2.0
    elif mode == "rank":
        # robust to calibration; uses cross-sectional rank
        r = np.argsort(np.argsort(p)).astype(float)
        s = r / max(1.0, (len(p) - 1.0))
    elif mode == "sigmoid":
        # squashes extremes, useful if p isn't calibrated
        s = 1.0 / (1.0 + np.exp(-sigmoid_k * (p - 0.5)))
    else:
        raise ValueError(f"Unknown position mode: {mode}")

    # Final scale to [min_pos, max_pos]
    s = np.clip(s, 0.0, 1.0) if (min_pos >= 0.0 and max_pos >= 0.0) else np.clip(s, 0.0, 1.0)
    pos = min_pos + (max_pos - min_pos) * s
    return pos


def calculate_selection_score(
    y_true,
    y_prob,
    trade_returns,
    *,
    sample_weight=None,
    # ---- Position sizing ----
    size_mode="rank",       # "linear"|"rank"|"sigmoid"|"centered"
    min_pos=0.0,              # long-only default. set -1..1 for long/short with centered mode
    max_pos=1.0,
    size_center=0.5,          # centered mode
    sigmoid_k=6.0,            # sigmoid mode
    size_clip=(0.0, 1.0),     # additional clip safety
    leverage=3.0,             # scalar multiplier on position size

    # ---- Realized metric ----
    cost_per_trade=0.005,       # cost per unit position (so pos*cost). Default set to 0.5%
    use_log_equity=False,
    annualization_factor=None,
    dd_penalty=0.25,
    coverage_penalty=0.10,

    # ---- Utility-weighted IC ----
    utility_clip=3.0,
    utility_power=1.0,
    ic_cap=0.10,

    # ---- BSS ----
    bss_min_prev=0.02,
    bss_cap_ref_min=1e-6,

    # ---- Composite weights ----
    w_realized=0.55,
    w_uic=0.35,
    w_bss=0.10,
):
    """
    Same as v2, but realized returns are *position-sized*:
        sized_return_i = position_i * trade_return_i - abs(position_i)*cost_per_trade

    Position is derived from y_prob (or its rank), enabling bet sizing / leverage effects to
    influence the realized metric and the utility-weighted IC.
    """
    y_true = np.asarray(y_true) if y_true is not None else None
    y_prob = np.asarray(y_prob, dtype=float)
    r = np.asarray(trade_returns, dtype=float)

    n = min(len(y_prob), len(r), len(y_true) if y_true is not None else len(y_prob))
    y_prob = y_prob[:n]
    r = r[:n]
    if y_true is not None:
        y_true = y_true[:n]

    m = np.isfinite(y_prob) & np.isfinite(r)
    if y_true is not None:
        m = m & np.isfinite(y_true)

    y_prob_m = y_prob[m]
    r_m = r[m]
    y_true_m = (y_true[m] >= 0.5).astype(int) if y_true is not None else None

    # Handle sample weights
    w_m = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        # align length first
        sample_weight = sample_weight[:n]
        w_m = sample_weight[m]

    metrics = {"N": int(n), "N_valid": int(m.sum())}
    if metrics["N_valid"] < 5:
        metrics.update({
            "Position_Mean": 0.0,
            "Position_AbsMean": 0.0,
            "Realized_Metric": 0.0,
            "Realized_Score": 0.0,
            "Utility_IC": 0.0,
            "Utility_IC_Score": 0.0,
            "BSS": 0.0,
            "BSS_Score": 0.5,
            "Selection_Score": 0.0,
        })
        return metrics

    # -------------------------
    # 0) Position sizing
    # -------------------------
    pos = _position_from_prob(
        y_prob_m,
        mode=size_mode,
        min_pos=min_pos,
        max_pos=max_pos,
        center=size_center,
        sigmoid_k=sigmoid_k,
    )
    pos = np.asarray(pos, dtype=float) * float(leverage)
    pos = np.clip(pos, float(size_clip[0]), float(size_clip[1])) if size_clip is not None else pos

    # -------------------------
    # 1) Position-sized realized returns
    # -------------------------
    # Cost scales with absolute exposure
    sized_r = pos * r_m - np.abs(pos) * float(cost_per_trade)

    # Equity curve
    if use_log_equity:
        equity = np.nancumsum(sized_r)
        peak = np.maximum.accumulate(equity)
        # DD as % from peak (assuming returns are log-returns approx)
        # 1 - exp(equity - peak)
        dd_series = 1.0 - np.exp(equity - peak)
    else:
        equity = np.nancumprod(1.0 + sized_r)
        peak = np.maximum.accumulate(equity)
        dd_series = (peak - equity) / np.maximum(peak, 1e-12)

    mu = float(np.nanmean(sized_r))
    sd = float(np.nanstd(sized_r, ddof=1)) if len(sized_r) > 1 else 0.0
    sharpe_per_trade = mu / (sd + 1e-12)
    sharpe = sharpe_per_trade * np.sqrt(float(annualization_factor)) if annualization_factor is not None else sharpe_per_trade

    mdd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
    # MDD Penalty: directly penalize % DD. 
    # e.g. 20% DD => 0.2 penalty. 
    # We want robust score in [0,1].
    
    # Coverage: Fraction of ACTIVE trades (abs(pos) > epsilon)
    active_mask = np.abs(pos) > 1e-6
    coverage = np.mean(active_mask)

    # Score components
    # Map Sharpe to [0,1] robustly (softsign centered at 0?)
    # softsign(x) = x / (1+|x|). Maps -inf->-1, inf->1.
    # We want 0->0.5? Or just positive sharpe focus?
    # Let's use user's softsign logic but clearer:
    # realized_raw in [-1, 1]
    
    # cov_term: penalize low coverage.
    cov_term = np.clip(coverage, 0.0, 1.0)
    
    # Penalty logic: 
    # Score = (Softsign(Sharpe) - P_dd * MDD + P_cov * Coverage) normalized?
    # Let's keep it simple:
    # Base = 0.5 + 0.5 * softsign(Sharpe)  (in 0..1)
    # Penalties subtraction
    
    base_realized = 0.5 * (1.0 + (sharpe / (1.0 + abs(sharpe))))
    
    # Penalize MDD: limit impact to say 0.3
    # If MDD=0.2 (20%), penalty = 0.2 * dd_penalty
    dd_impact = dd_penalty * mdd_pct
    
    # Reward coverage: small bump if coverage is high, or penalty if low?
    # User had coverage_penalty * cov_term (additive).
    # Let's say we want at least 5% coverage. 
    # If coverage < 0.05 => penalty.
    # Simpler: just add weighted coverage.
    cov_impact = coverage_penalty * cov_term

    realized_score = np.clip(base_realized - dd_impact + cov_impact, 0.0, 1.0)

    metrics["Position_Mean"] = float(np.nanmean(pos))
    metrics["Position_AbsMean"] = float(np.nanmean(np.abs(pos)))
    metrics["Sized_Return_Mean"] = mu
    metrics["Sized_Return_Std"] = sd
    metrics["Sharpe"] = float(sharpe)
    metrics["Max_Drawdown"] = mdd_pct
    metrics["Coverage"] = float(coverage)
    metrics["Realized_Score"] = float(realized_score)

    # -------------------------
    # 2) Utility-weighted IC (prob vs utility of *UNIT* returns)
    # -------------------------
    # Avoid feedback loop: Usage of Sized returns inflates IC.
    # Use r_m (raw unit trade returns) for utility calculation.
    ur = _zscore(r_m)
    ur = np.clip(ur, -float(utility_clip), float(utility_clip))
    # Utility function: focuses on tails of the *market opportunity*
    u = np.sign(ur) * (np.abs(ur) ** float(utility_power))

    if np.nanstd(y_prob_m) < 1e-12 or np.nanstd(u) < 1e-12:
        uic = 0.0
    else:
        uic = spearmanr(y_prob_m, u, nan_policy="omit").correlation
        uic = 0.0 if (uic is None or not np.isfinite(uic)) else float(uic)

    # Sigmoid scaling for IC to prevent hard cap saturation
    # sigmoid: 2 / (1 + exp(-x/s)) - 1
    # scale s ~ 0.05 so that IC=0.10 => score ~ 0.76, IC=0.2 => score ~ 0.96
    s_ic = 0.08
    uic_score = 2.0 / (1.0 + np.exp(-max(0.0, uic) / s_ic)) - 1.0
    uic_score = np.clip(uic_score, 0.0, 1.0)

    metrics["Utility_IC"] = float(uic)
    metrics["Utility_IC_Score"] = float(uic_score)

    # -------------------------
    # 3) Brier Skill Score (calibration)
    # -------------------------
    # IMPORTANT: BSS is computed WITHOUT sample weights.
    # Sample weights upweight minority class for training loss, but BSS must
    # measure calibration on the actual data distribution. Using weights makes
    # weighted_prev ≈ 0.5 even when actual_prev ≈ 0.31, causing BS_ref ≈ 0.25
    # and negative BSS even for models with AUC > 0.5.
    bss = 0.0
    bss_score = 0.5
    bs = 0.0
    bs_ref = 0.0
    brier_basic = 0.0
    
    if y_true_m is not None:
        p = np.clip(y_prob_m, 0.0, 1.0)
        
        # UNWEIGHTED prevalence for BSS reference
        prev = float(np.mean(y_true_m)) if len(y_true_m) else 0.0

        # Basic (unweighted) Brier score — always computed
        try:
            brier_basic = float(brier_score_loss(y_true_m, p))
        except Exception:
            brier_basic = 0.0
            
        if bss_min_prev < prev < (1.0 - bss_min_prev):
            try:
                # UNWEIGHTED Brier scores for BSS
                bs = float(brier_score_loss(y_true_m, p))
                bs_ref = float(brier_score_loss(y_true_m, np.full_like(p, prev)))
                    
                bs_ref = max(bs_ref, float(bss_cap_ref_min))
                bss = 1.0 - (bs / bs_ref)
                if not np.isfinite(bss):
                    bss = 0.0
                bss_score = _normalize_bss(bss)
            except Exception:
                bss, bss_score = 0.0, 0.5

    metrics["BSS"] = float(bss)
    metrics["BSS_Score"] = float(bss_score)
    # Add raw components for diagnostics
    metrics["Brier_Score"] = float(bs) if y_true_m is not None else 0.0
    metrics["Brier_Ref"] = float(bs_ref) if y_true_m is not None else 0.0
    metrics["Brier"] = float(brier_basic) if y_true_m is not None else 0.0

    # -------------------------
    # 4) Top-K Precision
    # -------------------------
    if y_true_m is not None and len(y_true_m) > 10:
        # Sort by prob descending
        idx_sorted = np.argsort(y_prob_m)[::-1]
        y_sorted = y_true_m[idx_sorted]
        w_sorted = w_m[idx_sorted] if w_m is not None else np.ones_like(y_sorted)
        
        # Top 10%
        n_10 = max(1, int(len(y_sorted) * 0.10))
        prec_10 = np.average(y_sorted[:n_10], weights=w_sorted[:n_10])
        metrics["Prec_Top10"] = float(prec_10)
        
        # Top 25%
        n_25 = max(1, int(len(y_sorted) * 0.25))
        prec_25 = np.average(y_sorted[:n_25], weights=w_sorted[:n_25])
        metrics["Prec_Top25"] = float(prec_25)

        # Top 40%
        n_40 = max(1, int(len(y_sorted) * 0.40))
        prec_40 = np.average(y_sorted[:n_40], weights=w_sorted[:n_40])
        metrics["Prec_Top40"] = float(prec_40)
    else:
        metrics["Prec_Top10"] = 0.0
        metrics["Prec_Top25"] = 0.0
        metrics["Prec_Top40"] = 0.0

    # -------------------------
    # 5) Composite
    # -------------------------
    # Adjust weights to de-emphasize BSS if using Rank sizing
    # E.g. 0.60, 0.35, 0.05
    sel = (
        float(w_realized) * metrics["Realized_Score"] +
        float(w_uic) * metrics["Utility_IC_Score"] +
        float(w_bss) * metrics["BSS_Score"]
    )
    metrics["Selection_Score"] = float(np.clip(sel, 0.0, 1.0))

    # Diagnostic keys
    if y_true_m is not None:
        try:
             if len(np.unique(y_true_m)) > 1:
                 metrics["AUC"] = float(roc_auc_score(y_true_m, y_prob_m))
             else:
                 metrics["AUC"] = 0.5
        except:
             metrics["AUC"] = 0.5
    else:
        metrics["AUC"] = 0.5
        
    metrics["IC"] = metrics["Utility_IC"]
    # Or keep original IC? The new logic uses Utility IC.
    # We can add standard IC too.
    try:
        std_ic = spearmanr(y_prob_m, r_m, nan_policy="omit").correlation
        metrics["Standard_IC"] = float(std_ic) if (std_ic is not None and np.isfinite(std_ic)) else 0.0
    except:
        metrics["Standard_IC"] = 0.0
    
    return metrics
