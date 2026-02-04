import numpy as np
from .utils import tprint

EPS = 1e-12

# -----------------------------
# Hard constraints
# -----------------------------
def apply_hard_constraints(raw_pos,
                           vol=None,
                           p_max=1.0,
                           target_vol=None,
                           vol_floor=1e-8):
    """
    Hard exposure cap + optional hard volatility targeting.

    raw_pos: (T,) unconstrained real-valued position signal
    vol: (T,) realized vol estimate aligned to periods (same length as raw_pos)
    p_max: hard cap on |position|
    target_vol: if not None and vol is provided, scale position by target_vol / vol
    vol_floor: avoid division blow-ups when vol ~ 0
    """
    tprint(f"Entering function: apply_hard_constraints in optimization.py")
    pos = np.asarray(raw_pos, dtype=float).copy()

    # Hard volatility targeting (risk scaling)
    if (target_vol is not None) and (vol is not None):
        v = np.maximum(np.asarray(vol, dtype=float), vol_floor)
        pos = pos * (target_vol / v)
        tprint(f"  Vol scaling applied. Mean pos: {np.mean(pos):.6f}, Max pos: {np.max(pos):.6f}")

    # Hard exposure cap
    pos = np.clip(pos, -p_max, p_max)
    tprint(f"  Hard constraints applied. Mean pos: {np.mean(pos):.6f}, Max pos: {np.max(pos):.6f}")
    return pos


# -----------------------------
# Trading metrics
# -----------------------------
def pnl_series(returns, position, cost_per_turnover=0.0):
    """
    PnL_t = position_t * return_t - cost * |position_t - position_{t-1}|
    """
    tprint(f"Entering function: pnl_series in optimization.py")
    r = np.asarray(returns, dtype=float)
    p = np.asarray(position, dtype=float)
    turnover = np.abs(np.diff(p, prepend=p[0]))
    tprint(f"  Calculated turnover. Mean turnover: {np.mean(turnover):.6f}")
    return p * r - cost_per_turnover * turnover

def total_pnl(pnl):
    tprint(f"Entering function: total_pnl in optimization.py")
    return float(np.nansum(pnl))

def sortino_ratio(pnl, annualization_factor=252):
    tprint(f"Entering function: sortino_ratio in optimization.py")
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    mu = np.mean(x)
    downside = np.minimum(x, 0.0)
    dd = np.sqrt(np.mean(downside**2) + EPS)
    tprint(f"  Sortino components: mu={mu:.6f}, dd={dd:.6f}")
    return float((mu / dd) * np.sqrt(annualization_factor))

def max_drawdown(pnl):
    tprint(f"Entering function: max_drawdown in optimization.py")
    x = np.asarray(pnl, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    tprint(f"  Max drawdown calculated: {np.max(dd):.6f}")
    return float(np.max(dd))

def robust_zscore(value, baseline_values):
    """
    Robust scaling: (value - median) / IQR
    """
    tprint(f"Entering function: robust_zscore in optimization.py")
    b = np.asarray(baseline_values, dtype=float)
    b = b[np.isfinite(b)]
    if b.size < 4:
        mu = np.mean(b) if b.size else 0.0
        sd = np.std(b) + EPS
        tprint(f"  Baseline size < 4. Using mean/std. mu={mu:.6f}, sd={sd:.6f}")
        return float((value - mu) / sd)
    med = np.median(b)
    q75, q25 = np.percentile(b, [75, 25])
    iqr = (q75 - q25) + EPS
    tprint(f"  Robust stats. Median={med:.6f}, IQR={iqr:.6f}")
    return float((value - med) / iqr)


# -----------------------------
# Composite objective (with hard constraints)
# -----------------------------
def composite_score_with_constraints(
    returns,
    raw_position,
    vol=None,
    # hard constraints
    p_max=1.0,
    target_vol=None,
    vol_floor=1e-8,
    # costs + annualization
    cost_per_turnover=0.0,
    annualization_factor=252,
    # optional baseline distributions for normalization
    baseline_pnls=None,
    baseline_sortinos=None,
    baseline_maxdds=None,
    # weights
    w_pnl=0.6,
    w_sortino=0.3,
    w_maxdd=0.1,
):
    """
    Applies hard exposure and (optional) hard vol-targeting, then computes:

      Score = 0.6*norm(PnL) + 0.3*norm(Sortino) - 0.1*norm(MaxDD)

    If baselines are None, uses raw metrics (less recommended).
    """
    tprint(f"Entering function: composite_score_with_constraints in optimization.py")
    pos = apply_hard_constraints(
        raw_position, vol=vol, p_max=p_max, target_vol=target_vol, vol_floor=vol_floor
    )

    pnl = pnl_series(returns, pos, cost_per_turnover=cost_per_turnover)

    m_pnl = total_pnl(pnl)
    m_sort = sortino_ratio(pnl, annualization_factor=annualization_factor)
    m_mdd = max_drawdown(pnl)

    tprint(f"  Metrics calculated: PnL={m_pnl:.4f}, Sortino={m_sort:.4f}, MaxDD={m_mdd:.4f}")

    m_pnl_n = robust_zscore(m_pnl, baseline_pnls) if baseline_pnls is not None else m_pnl
    m_sort_n = robust_zscore(m_sort, baseline_sortinos) if baseline_sortinos is not None else m_sort
    m_mdd_n = robust_zscore(m_mdd, baseline_maxdds) if baseline_maxdds is not None else m_mdd

    tprint(f"  Normalized metrics: PnL_n={m_pnl_n:.4f}, Sortino_n={m_sort_n:.4f}, MaxDD_n={m_mdd_n:.4f}")

    score = w_pnl * m_pnl_n + w_sortino * m_sort_n - w_maxdd * m_mdd_n
    tprint(f"  Composite score: {score:.4f}")

    metrics = {
        "PnL": m_pnl,
        "Sortino": m_sort,
        "MaxDD": m_mdd,
        "PnL_norm": m_pnl_n,
        "Sortino_norm": m_sort_n,
        "MaxDD_norm": m_mdd_n,
        "Score": float(score),
        "mean_abs_position": float(np.mean(np.abs(pos))),
        "position_min": float(np.min(pos)),
        "position_max": float(np.max(pos)),
    }
    return float(score), metrics
