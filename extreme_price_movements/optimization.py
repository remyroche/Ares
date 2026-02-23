import numpy as np
from dataclasses import dataclass
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

    # Hard exposure cap
    pos = np.clip(pos, -p_max, p_max)
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
    return float((mu / dd) * np.sqrt(annualization_factor))

def max_drawdown(pnl):
    tprint(f"Entering function: max_drawdown in optimization.py")
    x = np.asarray(pnl, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
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
        return float((value - mu) / sd)
    med = np.median(b)
    q75, q25 = np.percentile(b, [75, 25])
    iqr = (q75 - q25) + EPS
    return float((value - med) / iqr)


def equity_curve_from_returns(r: np.ndarray, equity0: float = 1.0) -> np.ndarray:
    """Build compounded equity curve from per-period returns."""
    r = np.asarray(r, dtype=float)
    if np.any(r <= -1.0):
        raise ValueError("r_t <= -1 found; log-growth undefined.")
    return float(equity0) * np.cumprod(1.0 + r)


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Drawdown series in [0, 1)."""
    equity = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peaks, EPS)
    return np.clip(dd, 0.0, 1.0 - 1e-12)


def ulcer_index(dd: np.ndarray) -> float:
    dd = np.asarray(dd, dtype=float)
    return float(np.sqrt(np.mean(dd * dd))) if dd.size else 0.0


def recovery_episodes(dd: np.ndarray):
    """Return arrays of episode depth, duration, and recovery flags."""
    dd = np.asarray(dd, dtype=float)
    T = len(dd)

    depths, durations, recovered = [], [], []
    in_ep = False
    start = 0
    max_dd = 0.0

    for t in range(T):
        if not in_ep:
            if dd[t] > 0.0:
                in_ep = True
                start = t
                max_dd = dd[t]
        else:
            max_dd = max(max_dd, dd[t])
            if dd[t] <= 0.0:
                in_ep = False
                depths.append(max_dd)
                durations.append(t - start + 1)
                recovered.append(True)

    if in_ep:
        depths.append(max_dd)
        durations.append(T - start)
        recovered.append(False)

    if not depths:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)
    return np.asarray(depths, dtype=float), np.asarray(durations, dtype=float), np.asarray(recovered, dtype=bool)


def expected_recovery_speed(dd: np.ndarray, eps: float = 1.0, use_log: bool = True):
    """Expected recovery speed and unrecovered-episode probability."""
    A, D, rec = recovery_episodes(dd)
    if A.size == 0:
        return 0.0, 0.0
    ratio = A / (D + eps)
    rs = float(np.mean(np.log1p(ratio))) if use_log else float(np.mean(ratio))
    p_not_rec = float(np.mean(~rec))
    return rs, p_not_rec


@dataclass
class RiskBudgetConfig:
    ui_max: float
    x_min: float
    lambda_rs: float = 0.10
    eps_recovery_hours: float = 1.0
    use_log_rs: bool = True
    penalize_not_recovered: float = 0.0
    equity0: float = 1.0
    hard_fail: bool = True
    soft_penalty_scale: float = 50.0


def score_backtest_risk_budgeted(r: np.ndarray, x: np.ndarray, cfg: RiskBudgetConfig):
    """Risk-budgeted backtest score for optimise/grid/Optuna."""
    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(r) != len(x):
        raise ValueError("r and x must have same length.")

    xbar = float(np.mean(np.abs(x))) if x.size else 0.0
    G = float(np.mean(np.log1p(r))) if r.size else 0.0
    E = equity_curve_from_returns(r, equity0=cfg.equity0)
    dd = drawdown_series(E)
    UI = ulcer_index(dd)
    RS, p_not_rec = expected_recovery_speed(dd, eps=cfg.eps_recovery_hours, use_log=cfg.use_log_rs)

    ui_violation = max(0.0, UI - cfg.ui_max)
    x_violation = max(0.0, cfg.x_min - xbar)

    if cfg.hard_fail and (ui_violation > 0.0 or x_violation > 0.0):
        score = -1e9
    else:
        score = G + cfg.lambda_rs * RS - cfg.penalize_not_recovered * p_not_rec
        if not cfg.hard_fail:
            score -= cfg.soft_penalty_scale * (ui_violation**2 + x_violation**2)

    return {
        "score": float(score),
        "G_mean_log1p": G,
        "UlcerIndex": UI,
        "RecoverySpeed": RS,
        "p_not_recovered": p_not_rec,
        "xbar": xbar,
        "ui_violation": ui_violation,
        "x_violation": x_violation,
    }


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

    m_pnl_n = robust_zscore(m_pnl, baseline_pnls) if baseline_pnls is not None else m_pnl
    m_sort_n = robust_zscore(m_sort, baseline_sortinos) if baseline_sortinos is not None else m_sort
    m_mdd_n = robust_zscore(m_mdd, baseline_maxdds) if baseline_maxdds is not None else m_mdd

    score = w_pnl * m_pnl_n + w_sortino * m_sort_n - w_maxdd * m_mdd_n

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
