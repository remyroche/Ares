"""Policy optimisation stage for extreme_price_movements.

This module consumes the already-selected best strategy (from simple_position_sizer),
keeps TP/SL fixed, runs offset generation first, then sequentially optimises richer
exit-policy parameters. The resulting params are persisted for holdout OOS replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.run_ridge_sizer import (
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.simple_offset_generator import (
    build_policy_path_state_bundle,
    run_simple_offset_generator_from_sizer,
)
from extreme_price_movements.utils import tprint

EPS = 1e-9


def _metric_score(
    rets: np.ndarray, cost_pct: float = 0.003, already_net: bool = False
) -> Dict[str, float]:
    if len(rets) == 0:
        return {
            "net_pnl": 0.0,
            "sortino": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
        }
    # Avoid double-counting costs if returns already include them
    if already_net:
        net_rets = rets.astype(np.float64)
    else:
        net_rets = rets.astype(np.float64) - float(cost_pct)
    downside = net_rets[net_rets < 0]
    ds_std = float(np.std(downside)) if len(downside) > 1 else 1e-6
    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    _, dd = _stable_equity_and_drawdown(net_rets)
    return {
        "net_pnl": float(np.sum(net_rets)),
        "sortino": float(np.mean(net_rets)) / ds_std,
        "hit_rate": float(np.mean(net_rets > 0)),
        "profit_factor": gross_win / gross_loss
        if gross_loss > EPS
        else float(gross_win),
        "max_drawdown": float(np.max(dd)) if len(dd) else 0.0,
        "n_trades": int(len(rets)),
    }


def _robust_fit(arr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(arr, dtype=np.float64)
    tr = x[mask]
    if tr.size == 0:
        return 0.0, 1.0, -5.0, 5.0
    med = float(np.nanmedian(tr))
    q1 = float(np.nanpercentile(tr, 25))
    q3 = float(np.nanpercentile(tr, 75))
    iqr = max(q3 - q1, 1e-6)
    lo = float(np.nanpercentile((tr - med) / iqr, 1))
    hi = float(np.nanpercentile((tr - med) / iqr, 99))
    return med, iqr, lo, hi


def _robust_apply(
    arr: np.ndarray, params: Tuple[float, float, float, float]
) -> np.ndarray:
    med, iqr, lo, hi = params
    z = (np.asarray(arr, dtype=np.float64) - med) / max(iqr, 1e-6)
    return np.clip(z, lo, hi).astype(np.float32)


def _load_best_strategy(data_root: str, run_id: str) -> Dict[str, Any]:
    """Load the post head-to-head winner from sizer artifacts.

    Preference order:
      1) explicit winner from ridge_sizer/head_to_head_comparison.json
      2) best_strategy_id from et/ridge strategy_params.json (higher net_pnl)
    """
    comparison_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "head_to_head_comparison.json"
    )
    if comparison_path.exists():
        try:
            rows = json.loads(comparison_path.read_text())
            et_winners = [r for r in rows if str(r.get("winner", "")).lower() == "et"]
            if et_winners:
                et_params = (
                    Path(data_root)
                    / "artifacts"
                    / run_id
                    / "et_sizer"
                    / "strategy_params.json"
                )
                if et_params.exists():
                    payload = json.loads(et_params.read_text())
                    sid = str(payload.get("best_strategy_id", ""))
                    bucket = (payload.get("buckets", {}) or {}).get(sid, {})
                    return {
                        "strategy_id": sid,
                        "threshold_pct": float(
                            bucket.get(
                                "threshold_pct", payload.get("best_threshold_pct", 90.0)
                            )
                        ),
                        "model": "et",
                        "tp_mult": float(bucket.get("tp_mult", 1.0)),
                        "sl_mult": float(bucket.get("sl_mult", 1.0)),
                    }
        except Exception:
            pass

    candidates = [
        Path(data_root) / "artifacts" / run_id / "et_sizer" / "strategy_params.json",
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json",
    ]
    best: Dict[str, Any] = {}
    best_pnl = -1e18
    for pth in candidates:
        if not pth.exists():
            continue
        payload = json.loads(pth.read_text())
        buckets = payload.get("buckets", {})
        sid = str(payload.get("best_strategy_id", ""))
        row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
        pnl = float(row.get("net_pnl", -1e18))
        if pnl > best_pnl:
            best_pnl = pnl
            best = {
                "strategy_id": sid,
                "threshold_pct": float(
                    row.get("threshold_pct", payload.get("best_threshold_pct", 90.0))
                ),
                "model": "et" if "et_sizer" in str(pth) else "ridge",
                "tp_mult": float(row.get("tp_mult", 1.0)),
                "sl_mult": float(row.get("sl_mult", 1.0)),
            }
    return best


def resolve_optimised_selection_frac(
    *, data_root: str, run_id: str, selected: Dict[str, Any]
) -> float:
    """Return selection fraction from ranked threshold with optimised pnl.

    Uses the selected strategy threshold from `strategy_params.json` generated by
    simple_position_sizer (which stores the optimal ranked threshold). If the selected
    row has non-positive pnl, fallback to the highest positive-pnl threshold in the
    same model artifact.
    """
    model = str(selected.get("model", "ridge")).lower()
    params_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / ("et_sizer" if model == "et" else "ridge_sizer")
        / "strategy_params.json"
    )
    threshold_pct = float(selected.get("threshold_pct", 90.0))
    if params_path.exists():
        try:
            payload = json.loads(params_path.read_text())
            buckets = payload.get("buckets", {}) or {}
            sid = str(selected.get("strategy_id", ""))
            row = buckets.get(sid, {}) if isinstance(buckets, dict) else {}
            row_pnl = float(row.get("net_pnl", 0.0))
            threshold_pct = float(row.get("threshold_pct", threshold_pct))
            if row_pnl <= 0.0 and isinstance(buckets, dict):
                positive_rows = [
                    r for r in buckets.values() if float(r.get("net_pnl", -1e18)) > 0.0
                ]
                if positive_rows:
                    best_pos = max(
                        positive_rows, key=lambda r: float(r.get("net_pnl", -1e18))
                    )
                    threshold_pct = float(best_pos.get("threshold_pct", threshold_pct))
        except Exception:
            pass

    frac = max(0.01, min(1.0, 1.0 - threshold_pct / 100.0))
    return float(frac)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def build_replay_context(
    *,
    returns: np.ndarray,
    mfe_ret: np.ndarray,
    mae_ret: np.ndarray,
    bars_since_entry: np.ndarray,
    barrier_pct: np.ndarray,
    confidence: np.ndarray,
    trend: Optional[np.ndarray] = None,
    choppiness: Optional[np.ndarray] = None,
    asym: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build canonical replay context shared by optimiser and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32)
    mfe = np.asarray(mfe_ret, dtype=np.float32)
    mae = np.asarray(mae_ret, dtype=np.float32)
    bars = np.maximum(1, np.asarray(bars_since_entry, dtype=np.int32))
    barr = np.maximum(np.asarray(barrier_pct, dtype=np.float32), 1e-4)
    conf = np.asarray(confidence, dtype=np.float32)

    ae_vel = mae / np.maximum(bars, 1)
    pressure = mae / (mfe + EPS)
    path_quality = mfe - mae
    progress_per_bar = np.maximum(rets, 0.0) / np.maximum(barr * bars, 1e-4)

    asym_raw = np.log((mfe + 1e-6) / (mae + 1e-6))
    if asym is None:
        asym_vals = asym_raw
    else:
        asym_vals = np.asarray(asym, dtype=np.float32)

    return {
        "mfe_ret": mfe,
        "mae_ret": mae,
        "bars_since_entry": bars,
        "barrier_pct": barr,
        "confidence": conf,
        "AE_vel": ae_vel.astype(np.float32),
        "pressure": pressure.astype(np.float32),
        "path_quality": path_quality.astype(np.float32),
        "progress_per_bar": progress_per_bar.astype(np.float32),
        "trend": np.asarray(
            np.zeros_like(rets) if trend is None else trend, dtype=np.float32
        ),
        "choppiness": np.asarray(
            np.zeros_like(rets) if choppiness is None else choppiness, dtype=np.float32
        ),
        "asym": np.asarray(asym_vals, dtype=np.float32),
        "asym_raw": asym_raw.astype(np.float32),
    }


def replay_exit_policy(
    returns: np.ndarray, context: Dict[str, np.ndarray], params: Dict[str, Any]
) -> np.ndarray:
    """Shared replay logic for optimisation and OOS evaluation."""
    rets = np.asarray(returns, dtype=np.float32).copy()
    mfe = np.asarray(context.get("mfe_ret", np.abs(rets)), dtype=np.float32)
    mae = np.asarray(
        context.get("mae_ret", np.abs(np.minimum(rets, 0.0))), dtype=np.float32
    )
    bars = np.maximum(
        1,
        np.asarray(
            context.get("bars_since_entry", np.full(len(rets), 4)), dtype=np.int32
        ),
    )
    conf = np.asarray(context.get("confidence", np.zeros(len(rets))), dtype=np.float32)
    barrier = np.maximum(
        np.asarray(
            context.get("barrier_pct", np.full(len(rets), 0.02)), dtype=np.float32
        ),
        1e-4,
    )

    tp_mult = float(params.get("tp_mult", 1.0))
    sl_mult = float(params.get("sl_mult", 1.0))
    tp_dist = tp_mult * barrier
    sl_dist = sl_mult * barrier

    ae_vel = np.asarray(
        context.get("AE_vel", mae / np.maximum(bars, 1)), dtype=np.float32
    )
    # NOTE: pressure uses cumulative formula here for backward compatibility,
    # but the policy_optimiser overwrites this with delta-based pressure from path_bundle.
    # Ensure OOS replay uses path_bundle-based context to get consistent delta-pressure.
    pressure = np.asarray(context.get("pressure", mae / (mfe + EPS)), dtype=np.float32)
    path_quality = np.asarray(context.get("path_quality", mfe - mae), dtype=np.float32)
    progress = np.asarray(
        context.get(
            "progress_per_bar", np.maximum(rets, 0.0) / np.maximum(bars * barrier, 1e-4)
        ),
        dtype=np.float32,
    )

    a1 = float(params.get("a1", 0.5))
    a2 = float(params.get("a2", 1.0 - a1))
    b1 = float(params.get("b1", 0.5))
    b2 = float(params.get("b2", 1.0 - b1))
    theta_fail = float(params.get("theta_fail", 0.0))
    theta_path = float(params.get("theta_path", 0.0))
    d_path = int(params.get("d_path", 1))
    k_early = int(params.get("K_early", 3))

    p_tp = np.asarray(context.get("p_tp", np.full(len(rets), np.nan)), dtype=np.float32)
    p_sl = np.asarray(context.get("p_sl", np.full(len(rets), np.nan)), dtype=np.float32)
    has_barrier_proba = np.any(np.isfinite(p_tp)) and np.any(p_tp > 0)

    trend = np.asarray(context.get("trend", np.zeros(len(rets))), dtype=np.float32)
    asym = np.asarray(context.get("asym", np.full(len(rets), 0.5)), dtype=np.float32)
    choppy = np.asarray(
        context.get("choppiness", np.zeros(len(rets))), dtype=np.float32
    )
    s = 1.0 * trend + 0.6 * asym - 0.9 * choppy
    m_raw = 0.7 + 0.6 * _sigmoid(s)
    m = np.clip(
        m_raw,
        float(params.get("multiplier_band_min", 0.8)),
        float(params.get("multiplier_band_max", 1.2)),
    )

    fail_scale = 1.0 - 0.5 * (m - 1.0)
    s_fail = (a1 * ae_vel + a2 * pressure) * fail_scale
    s_path = b1 * path_quality + b2 * progress

    tp_gate = float(params.get("tp_conf_threshold", 0.5))
    sl_gate = float(params.get("sl_early_exit_threshold", 0.4))

    if has_barrier_proba:
        p_sl_safe = np.where(np.isfinite(p_sl), p_sl, 0.5)
        fail_exit = (bars <= k_early) & (p_sl_safe >= sl_gate)
    else:
        fail_exit = (bars <= k_early) & (s_fail > theta_fail)

    path_exit = (
        (bars >= max(3, d_path))
        & (progress >= float(params.get("progress_threshold", 0.0)))
        & (s_path < theta_path)
    )
    rets = np.where(fail_exit | path_exit, np.minimum(rets, -0.25 * sl_dist), rets)

    mfe_norm = mfe / np.maximum(tp_dist, 1e-4)
    c_start = float(params.get("compression_start", 0.5))
    c_full_raw = float(params.get("compression_full", 1.0))
    if c_full_raw <= c_start:
        import warnings

        warnings.warn(
            f"compression_full ({c_full_raw}) <= compression_start ({c_start}). "
            f"Adjusting to compression_start + 1e-6. Review your parameter grid.",
            UserWarning,
        )
    c_full = max(c_start + 1e-6, c_full_raw)
    c_max = float(params.get("compression_max_fraction", 0.5))
    c_alpha = np.clip((mfe_norm - c_start) / (c_full - c_start), 0.0, 1.0) * c_max
    sl_eff = sl_dist * (1.0 - c_alpha)
    rets = np.maximum(rets, -sl_eff)

    trail_act = float(params.get("trail_activation_atr", 1.0)) * (1.0 + 0.8 * (m - 1.0))
    trail_gb = float(params.get("trail_giveback_atr", 0.5)) * (1.0 + 0.8 * (m - 1.0))
    trail_on = mfe >= trail_act * barrier
    trail_floor = mfe - trail_gb * barrier

    if has_barrier_proba:
        p_tp_safe = np.where(np.isfinite(p_tp), p_tp, 0.5)
        high_cont = p_tp_safe >= tp_gate
    else:
        cont_thr = float(params.get("continuation_conf_threshold", 0.5))
        high_cont = conf >= cont_thr

    with_tp = np.minimum(np.maximum(rets, trail_floor), tp_dist)
    no_tp = np.maximum(rets, trail_floor)
    rets = np.where(trail_on & high_cont, no_tp, np.where(trail_on, with_tp, rets))

    sl_scale = 1.0 + 0.4 * (m - 1.0)
    tp_scale = 1.0 + 1.0 * (m - 1.0)
    rets = np.clip(rets, -sl_dist * sl_scale, tp_dist * tp_scale)
    return rets.astype(np.float32)


def _sequential_optimise(
    base_returns: np.ndarray,
    context: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    fixed: Dict[str, Any],
    cost_pct: float,
) -> Dict[str, Any]:
    params = dict(fixed)
    search_plan: List[Tuple[str, List[Any]]] = [
        ("theta_fail", np.linspace(-1.0, 1.5, 11).tolist()),
        ("theta_path", np.linspace(-1.5, 1.0, 11).tolist()),
        ("d_path", [1, 2, 3]),
        ("progress_threshold", [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
        ("tp_conf_threshold", [0.3, 0.4, 0.5, 0.6, 0.7]),
        ("sl_early_exit_threshold", [0.2, 0.3, 0.4, 0.5]),
        ("compression_start", [0.30, 0.50, 0.70]),
        ("compression_full", [0.80, 1.00, 1.20]),
        ("compression_max_fraction", [0.20, 0.40, 0.60]),
        ("trail_activation_atr", [0.5, 0.8, 1.0, 1.2]),
        ("trail_giveback_atr", [0.2, 0.3, 0.4, 0.5]),
        ("continuation_conf_threshold", [0.4, 0.5, 0.6, 0.7]),
        ("multiplier_band", [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]),
        ("a1", np.linspace(0.0, 1.0, 11).tolist()),
        ("b1", np.linspace(0.0, 1.0, 11).tolist()),
    ]

    for name, grid in search_plan:
        best_metric = -1e18
        best_val: Any = grid[0]
        for cand in grid:
            trial = dict(params)
            if name == "multiplier_band":
                trial["multiplier_band_min"], trial["multiplier_band_max"] = cand
            elif name == "a1":
                trial["a1"] = float(cand)
                trial["a2"] = 1.0 - float(cand)
            elif name == "b1":
                trial["b1"] = float(cand)
                trial["b2"] = 1.0 - float(cand)
            else:
                trial[name] = cand
            rets = replay_exit_policy(base_returns, context, trial)
            score = _metric_score(
                rets[train_mask], cost_pct=cost_pct, already_net=True
            )["net_pnl"]
            if score > best_metric:
                best_metric = score
                best_val = cand
        if name == "multiplier_band":
            params["multiplier_band_min"], params["multiplier_band_max"] = best_val
        elif name == "a1":
            params["a1"] = float(best_val)
            params["a2"] = 1.0 - float(best_val)
        elif name == "b1":
            params["b1"] = float(best_val)
            params["b2"] = 1.0 - float(best_val)
        else:
            params[name] = best_val
    params["metrics_baseline"] = _metric_score(
        base_returns[val_mask], cost_pct=cost_pct, already_net=True
    )
    final_rets = replay_exit_policy(base_returns, context, params)
    params["metrics_final"] = _metric_score(
        final_rets[val_mask], cost_pct=cost_pct, already_net=True
    )
    return params


def run_policy_optimisation(
    data_root: str,
    run_id: str,
    sizer_results: Optional[Dict[str, Any]] = None,
    holdout_frac: float = 0.10,
    cost_pct: float = 0.003,
) -> Dict[str, Any]:
    tprint("POLICY OPTIMISER START")
    selected = _load_best_strategy(data_root, run_id)
    if not selected.get("strategy_id"):
        tprint("No selected strategy found; skipping policy optimisation.")
        return {}

    meta_oofs = load_meta_oof_predictions(data_root, run_id)
    key = next((k for k in meta_oofs.keys() if selected["strategy_id"] in k), None)
    if key is None:
        tprint(f"Strategy {selected['strategy_id']} not found in meta OOF.")
        return {}

    outcomes = load_trade_outcomes(data_root, run_id, meta_oofs[key])
    if outcomes.empty:
        return {}

    conf = np.asarray(
        outcomes.get("oof_u_hat", pd.Series(np.zeros(len(outcomes)))).values,
        dtype=np.float32,
    )
    frac = resolve_optimised_selection_frac(
        data_root=data_root,
        run_id=run_id,
        selected=selected,
    )
    k = max(1, int(len(conf) * frac))
    # Use stable sort to ensure deterministic selection
    selected_idx = np.argsort(conf, kind="stable")[-k:]

    sizer_stub = {
        "best_simple_score_": conf,
        "best_simple_score_name_": "Ridge_Head_Sizer",
        "ridge_profit_proxy_table_": pd.DataFrame(
            [
                {
                    "selection_frac": frac,
                    "wallet_min": 0.05,
                    "wallet_max": 0.15,
                    "sizing_mode": "linear",
                    "is_optimal": True,
                }
            ]
        ),
        "opt_rets_": np.asarray(
            outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
            dtype=np.float32,
        )[selected_idx],
        "opt_ts_": np.asarray(
            outcomes.get("timestamp", pd.Series(np.arange(len(outcomes)))).values
        )[selected_idx],
    }

    offset_result = run_simple_offset_generator_from_sizer(
        sizer_results=sizer_stub, trade_outcomes=outcomes, cost_pct=cost_pct
    )
    # CRITICAL: Use the same indices as the sizer stub to ensure alignment
    path_bundle = build_policy_path_state_bundle(outcomes, selected_idx=selected_idx)

    base_rets = np.asarray(
        outcomes.get("return", pd.Series(np.zeros(len(outcomes)))).values,
        dtype=np.float32,
    )[offset_result.get("above_threshold_idx")]
    base_rets = np.where(
        offset_result.get("executed", np.ones(len(base_rets), dtype=bool)),
        base_rets,
        0.0,
    )

    ts = pd.to_datetime(
        np.asarray(path_bundle.get("timestamps", np.arange(len(base_rets)))),
        utc=True,
        errors="coerce",
    )
    order = np.argsort(ts.view("int64"))
    n = len(base_rets)
    n_val = max(1, int(n * holdout_frac))
    val_mask = np.zeros(n, dtype=bool)
    val_mask[order[-n_val:]] = True
    train_mask = ~val_mask

    ae_fit = _robust_fit(path_bundle["AE_vel"], train_mask)
    pr_fit = _robust_fit(path_bundle["pressure"], train_mask)
    pq_fit = _robust_fit(path_bundle["path_quality"], train_mask)
    pg_fit = _robust_fit(path_bundle["progress_per_bar"], train_mask)
    tr_fit = _robust_fit(path_bundle["trend"], train_mask)
    as_fit = _robust_fit(path_bundle["asym_raw"], train_mask)
    ch_fit = _robust_fit(path_bundle["choppiness"], train_mask)

    asym_z = _robust_apply(path_bundle["asym_raw"], as_fit)
    asym_01 = (asym_z - float(np.nanmin(asym_z[train_mask]))) / (
        float(np.nanmax(asym_z[train_mask]) - np.nanmin(asym_z[train_mask])) + EPS
    )
    context = build_replay_context(
        returns=base_rets,
        mfe_ret=path_bundle["mfe_ret"],
        mae_ret=path_bundle["mae_ret"],
        bars_since_entry=path_bundle["bars_since_entry"],
        barrier_pct=path_bundle["barrier_pct"],
        confidence=path_bundle["confidence"],
        trend=_robust_apply(path_bundle["trend"], tr_fit),
        choppiness=_robust_apply(path_bundle["choppiness"], ch_fit),
        asym=np.clip(asym_01, 0.0, 1.0).astype(np.float32),
    )
    context["AE_vel"] = _robust_apply(path_bundle["AE_vel"], ae_fit)
    context["pressure"] = _robust_apply(path_bundle["pressure"], pr_fit)
    context["path_quality"] = _robust_apply(path_bundle["path_quality"], pq_fit)
    context["progress_per_bar"] = _robust_apply(path_bundle["progress_per_bar"], pg_fit)
    context["p_tp"] = path_bundle.get("p_tp", np.full(len(base_rets), np.nan))
    context["p_sl"] = path_bundle.get("p_sl", np.full(len(base_rets), np.nan))

    fixed = {
        "strategy_id": selected["strategy_id"],
        "tp_mult": float(selected.get("tp_mult", 1.0)),
        "sl_mult": float(selected.get("sl_mult", 1.0)),
        "k_recent": 3,
        "K_early": 3,
        "lambda_path": 1.0,
        "a1": 0.5,
        "a2": 0.5,
        "b1": 0.5,
        "b2": 0.5,
        "score_weight_trend": 1.0,
        "score_weight_asym": 0.6,
        "score_weight_choppiness": 0.9,
    }

    best = _sequential_optimise(
        base_rets, context, train_mask, val_mask, fixed=fixed, cost_pct=cost_pct
    )
    payload = {
        "schema_version": "v4",
        "generated_by": "policy_optimiser",
        "run_id": run_id,
        "cost_pct": float(cost_pct),
        "strategies": [best],
    }
    out_dir = Path(data_root) / "artifacts" / run_id / "policy_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_policy_params.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    (Path(data_root) / "artifacts" / run_id / "best_policy_params.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    tprint(f"POLICY OPTIMISER COMPLETE -> {path}")
    return payload
