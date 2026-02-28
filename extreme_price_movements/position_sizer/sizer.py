from dataclasses import dataclass

import numpy as np


@dataclass
class PositionSizerConfig:
    ev_threshold: float = 0.0
    trade_percentile_threshold: float = 0.90
    rank_exponent: float = 2.0
    size_k: float = 1.0
    max_position_size: float = 1.0
    risk_epsilon: float = 1e-6
    costs_mode: str = "included_in_labels"
    exp_win_quantile: float = 0.50
    risk_loss_quantile: float = 0.90
    p_min: float = 1e-3
    alpha_power: float = 2.0
    score_temperature: float = 0.7
    turnover_lambda: float = 0.0


def percentile_rank(value: float, universe):
    vals = np.asarray(universe, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    v = float(value)
    less = float(np.mean(vals < v))
    equal = float(np.mean(vals == v))
    return float(np.clip(less + 0.5 * equal, 0.0, 1.0))


def conviction_threshold_from_opportunities(opportunity_evs, trade_percentile_threshold: float = 0.90):
    evs = np.asarray(opportunity_evs, dtype=float)
    evs = evs[np.isfinite(evs)]
    if len(evs) == 0:
        return float("inf")
    q = float(np.clip(trade_percentile_threshold, 0.0, 1.0))
    return float(np.quantile(evs, q))


def sharpen_alpha_score(alpha_score: float, alpha_power: float = 2.0):
    a = float(alpha_score)
    p = float(max(alpha_power, 1e-8))
    return float(np.sign(a) * (abs(a) ** p))


def temperature_scale_score(alpha_score: float, score_temperature: float = 0.7):
    t = float(max(score_temperature, 1e-8))
    return float(np.tanh(float(alpha_score) / t))


def size_position(
    pwin_hat: float,
    qwin50_hat: float,
    qwin80_hat: float,
    qloss50_hat: float,
    qloss90_hat: float,
    cfg: PositionSizerConfig,
    costs: float = 0.0,
    direction: float = 1.0,
    opportunity_evs=None,
    alpha_score: float = 1.0,
):
    p = float(np.clip(pwin_hat, cfg.p_min, 1.0 - cfg.p_min))
    w_win = float(qwin80_hat if cfg.exp_win_quantile >= 0.8 else qwin50_hat)
    l_loss = float(qloss90_hat if cfg.risk_loss_quantile >= 0.9 else qloss50_hat)
    w_win = max(w_win, 0.0)
    l_loss = max(l_loss, 0.0)

    ev = p * w_win - (1.0 - p) * l_loss - float(costs)
    risk = max((1.0 - p) * l_loss, 0.0)

    ev_universe = opportunity_evs if opportunity_evs is not None else np.array([ev], dtype=float)
    rank = percentile_rank(ev, ev_universe)
    conviction_score = float(rank ** float(max(cfg.rank_exponent, 1e-8)))
    conviction_cutoff = conviction_threshold_from_opportunities(
        ev_universe,
        trade_percentile_threshold=cfg.trade_percentile_threshold,
    )
    allowed = bool((ev > float(cfg.ev_threshold)) and (rank >= float(cfg.trade_percentile_threshold)))

    alpha_sharp = sharpen_alpha_score(alpha_score, cfg.alpha_power)
    scaled_score = temperature_scale_score(alpha_sharp, cfg.score_temperature)
    signed_direction = float(np.sign(direction) if direction != 0 else np.sign(scaled_score))
    if signed_direction == 0:
        signed_direction = 1.0

    raw_size = cfg.size_k * conviction_score * ev / (risk + cfg.risk_epsilon)
    base_size = float(np.clip(raw_size, 0.0, cfg.max_position_size))
    w = signed_direction * base_size
    if not allowed:
        w = 0.0

    return {
        "trade_allowed": allowed,
        "size": float(w),
        "raw_size": float(raw_size),
        "EV": float(ev),
        "Risk": float(risk),
        "rank": float(rank),
        "conviction_score": float(conviction_score),
        "conviction_cutoff": float(conviction_cutoff),
        "pwin": p,
        "W": w_win,
        "L": l_loss,
        "alpha_sharp": float(alpha_sharp),
        "scaled_score": float(scaled_score),
    }


def size_positions_ranked(ev_hat, risk_hat, alpha_score, cfg: PositionSizerConfig, group_ids=None):
    ev_hat = np.asarray(ev_hat, dtype=float)
    risk_hat = np.asarray(risk_hat, dtype=float)
    alpha_score = np.asarray(alpha_score, dtype=float)
    n = len(ev_hat)
    if group_ids is None:
        group_ids = np.zeros(n, dtype=int)
    else:
        group_ids = np.asarray(group_ids)

    sizes = np.zeros(n, dtype=float)
    ranks = np.zeros(n, dtype=float)
    allowed = np.zeros(n, dtype=bool)

    for g in np.unique(group_ids):
        m = group_ids == g
        ev_g = ev_hat[m]
        risk_g = np.maximum(risk_hat[m], 0.0)
        alpha_g = alpha_score[m]

        order = np.argsort(ev_g)
        rank_g = np.empty(len(ev_g), dtype=float)
        rank_g[order] = (np.arange(len(ev_g), dtype=float) + 0.5) / max(len(ev_g), 1)
        conviction = rank_g ** float(max(cfg.rank_exponent, 1e-8))
        gate = rank_g >= float(cfg.trade_percentile_threshold)

        alpha_sharp = np.sign(alpha_g) * (np.abs(alpha_g) ** float(max(cfg.alpha_power, 1e-8)))
        scaled = np.tanh(alpha_sharp / float(max(cfg.score_temperature, 1e-8)))
        dirn = np.sign(scaled)
        dirn[dirn == 0.0] = 1.0

        raw = cfg.size_k * conviction * ev_g / (risk_g + cfg.risk_epsilon)
        clipped = np.clip(raw, 0.0, cfg.max_position_size)
        out = dirn * clipped
        out[~gate] = 0.0

        sizes[m] = out
        ranks[m] = rank_g
        allowed[m] = gate

    return {
        "size": sizes,
        "rank": ranks,
        "trade_allowed": allowed,
    }
