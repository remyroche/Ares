from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _sigmoid(x: float) -> float:
    z = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def flatten_bucket_policy(bucket_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(bucket_cfg, dict):
        return {}
    out = dict(bucket_cfg)
    for k in ("tp_sl", "loss_limiter", "profit_exit", "position_sizing"):
        v = bucket_cfg.get(k)
        if isinstance(v, dict):
            out.update(v)
    return out


def _get_metric(features: Dict[str, Any], key: str, default: float) -> float:
    v = features.get(key, default)
    try:
        vv = float(v)
    except Exception:
        vv = default
    if not np.isfinite(vv):
        return default
    return vv


def compute_entry_policy_decision(
    entry_px: float,
    atr_frac: float,
    score: float,
    bucket_cfg: Dict[str, Any] | None,
    features: Dict[str, Any] | None = None,
) -> Dict[str, float | bool]:
    cfg = flatten_bucket_policy(bucket_cfg)
    ep = cfg.get("entry_policy", {}) if isinstance(cfg.get("entry_policy", {}), dict) else {}
    model = ep.get("model", {}) if isinstance(ep.get("model", {}), dict) else {}
    obj = ep.get("objective", {}) if isinstance(ep.get("objective", {}), dict) else {}
    adapt = ep.get("adaptation", {}) if isinstance(ep.get("adaptation", {}), dict) else {}
    f = features or {}

    u_z = _get_metric(f, "u_hat_z", float(np.tanh(score)))
    mae_z = _get_metric(f, "mae_hat_z", float(np.clip(abs(np.tanh(score)), 0.0, 3.0)))
    mfe_z = _get_metric(f, "mfe_hat_z", float(np.tanh(max(score, 0.0))))
    dur_z = _get_metric(f, "dur_hat_z", 0.0)

    alpha0 = float(model.get("alpha0", 0.4))
    alpha_u = float(model.get("alpha_u", 0.25))
    alpha_mae = float(model.get("alpha_mae", 0.25))
    beta_delta = float(model.get("beta_delta", 0.4))
    a = float(obj.get("a", 1.0))
    lambda_risk = float(obj.get("lambda_risk", 1.0))
    c_atr = float(obj.get("c_atr", 0.3))
    min_eu = float(obj.get("min_expected_utility", 0.0))
    delta_grid = obj.get("delta_atr_grid", None)
    if not isinstance(delta_grid, list) or not delta_grid:
        delta_grid = [x * 0.25 for x in range(0, 13)]

    best_eu = -1e18
    best_delta = 0.0
    best_p = 0.0
    for d in delta_grid:
        d = float(d)
        pfill = _sigmoid(alpha0 + alpha_u * u_z - alpha_mae * mae_z - beta_delta * d)
        eu = pfill * (a * u_z + d - lambda_risk * mae_z - c_atr)
        if eu > best_eu:
            best_eu = eu
            best_delta = d
            best_p = pfill

    place = bool(best_eu >= min_eu)
    atr_frac = float(np.clip(atr_frac, 1e-4, 0.5))
    entry_px = float(max(entry_px, 1e-9))
    delta_price = best_delta * atr_frac * entry_px
    limit_offset_pct = float(np.clip(delta_price / entry_px, 0.0, 0.10))
    limit_offset_bps = limit_offset_pct * 10000.0

    q_sl = float(adapt.get("q_sl", 1.3))
    eta = float(adapt.get("eta_stop", 0.4))
    r_tp = float(adapt.get("r_tp", 1.5))
    stop_factor = float(np.clip(1.0 - eta * best_delta, 0.5, 1.0))
    sl_distance_atr_eff = float(q_sl * max(mae_z, 0.0) * stop_factor)
    tp_distance_atr_eff = float(r_tp * max(a * u_z + best_delta, 0.0))

    trail_base = float(cfg.get("trail_mult", 0.25))
    giveback_base = float(cfg.get("giveback_pct", 0.005))
    lock_amt_base = float(cfg.get("profit_lock_amount", 0.003))
    kill_c_base = float(cfg.get("kill_c", 0.005))
    hold_h_base = float(cfg.get("max_hold_hours", 24.0))

    trail_mult_k_delta = float(adapt.get("trail_mult_k_delta", -0.08))
    trail_mult_k_mfe = float(adapt.get("trail_mult_k_mfe", 0.06))
    giveback_k_delta = float(adapt.get("giveback_k_delta", 0.05))
    giveback_k_dur = float(adapt.get("giveback_k_dur", 0.04))
    lock_amt_k_u = float(adapt.get("lock_amt_k_u", 0.08))
    kill_c_k_mae = float(adapt.get("kill_c_k_mae", 0.08))
    hold_h_k_dur = float(adapt.get("hold_h_k_dur", 0.20))

    trail_mult_eff = float(np.clip(trail_base * (1.0 + trail_mult_k_delta * best_delta + trail_mult_k_mfe * mfe_z), 0.05, 1.2))
    giveback_pct_eff = float(np.clip(giveback_base * (1.0 + giveback_k_delta * best_delta + giveback_k_dur * dur_z), 0.001, 0.05))
    profit_lock_amount_eff = float(np.clip(lock_amt_base * (1.0 + lock_amt_k_u * u_z), 0.0005, 0.05))
    kill_c_eff = float(np.clip(kill_c_base * (1.0 + kill_c_k_mae * max(mae_z, 0.0)), 0.0001, 0.05))
    max_hold_hours_eff = float(np.clip(hold_h_base * (1.0 + hold_h_k_dur * dur_z), 4.0, 72.0))

    return {
        "u_hat_z": float(u_z),
        "mae_hat_z": float(mae_z),
        "mfe_hat_z": float(mfe_z),
        "dur_hat_z": float(dur_z),
        "delta_atr_star": float(best_delta),
        "delta_price_star": float(delta_price),
        "p_fill_star": float(best_p),
        "eu_star": float(best_eu),
        "place_order": place,
        "limit_offset_bps_dynamic": float(limit_offset_bps),
        "entry_px_fill": float(max(entry_px - delta_price, 1e-9)),
        "sl_distance_atr_eff": sl_distance_atr_eff,
        "tp_distance_atr_eff": tp_distance_atr_eff,
        "trail_mult_eff": trail_mult_eff,
        "giveback_pct_eff": giveback_pct_eff,
        "profit_lock_amount_eff": profit_lock_amount_eff,
        "kill_c_eff": kill_c_eff,
        "max_hold_hours_eff": max_hold_hours_eff,
    }

