from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Sequence, List
import math
import numpy as np
import pandas as pd

EPS = 1e-12
TOL = 1e-9
# Policy proxy when net-TP quantiles are unavailable in this contract.
TRADEABLE_TP_MIN = 0.015

CANON_BUCKETS = ["MR_long", "MR_short", "TF_long", "TF_short"]
CANON_H = [2, 4, 8]
CANON_CELLS = [f"{b}_H{h}" for b in CANON_BUCKETS for h in CANON_H]


@dataclass(frozen=True)
class ProdGates:
    n_min: int = 50
    bind_cell_min: float = 0.38
    bind_min: float = 0.50
    timeout_max: float = 0.60
    timeout_range_max: float = 0.50
    sl_to_tp_max: float = 3.0
    tp_hit_min_agg: float = 0.0  # optional explicit TP-density floor

    auc_min: float = 0.56
    auc_bound_min: float = 0.52
    tp_sep_min: float = 0.05
    ap_lift_min: float = 1.25

    tp_over_sl_min: float = 1.05

    tp_floor_bind_max_cell: float = 0.70
    tp_floor_bind_max_agg: float = 0.65
    sl_floor_bind_max_cell: Optional[float] = None
    floor_bind_auc_boost: float = 0.02  # require stronger AUC when floor-binding is elevated
    enforce_tradeable_tp_lo: bool = False
    tradeable_tp_min: float = TRADEABLE_TP_MIN


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _cell_key(bucket: str, horizon: int) -> str:
    return f"{bucket}_H{int(horizon)}"


def compute_floor_binding(tp_eff: np.ndarray, tp_lo: float, tol: float = TOL) -> float:
    if tp_eff is None or len(tp_eff) == 0:
        return float("nan")
    tp_eff = np.asarray(tp_eff, dtype=np.float64)
    m = np.isfinite(tp_eff)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(tp_eff[m] <= (float(tp_lo) + tol)))


def compute_prod_aligned_tp_params(
    atr_pct_samples: np.ndarray,
    fee_pct_total: float,
    horizon_scaling_fn: Callable[[int], float],
    worst_horizon: int = 2,
    q: float = 0.25,
    alpha: float = 0.45,
    margin_mult: float = 4.0,
    hard_min_tp: float = 0.02,
    inflate: float = 1.10,
    horizons: Sequence[int] = (2, 4, 8),
    h2_lower: float = 0.01,
    h2_upper: float = 0.04,
    q_grid: Sequence[float] = (0.50, 0.75, 0.90),
    alpha_grid: Sequence[float] = (0.6, 0.8, 1.0, 1.2),
) -> Dict[str, Any]:
    """Compute production-aligned TP centering params and a TP-base candidate ladder.

    Candidate generation follows the H2/H4/H8 centering rule:
      tp_center_H2 = alpha * atr_q
      tp_eff_H2 = clip(tp_center_H2, [h2_lower, h2_upper])
      tp_base_pct = tp_eff_H2 / s(H2)
    with per-horizon implied diagnostics.
    """
    arr = np.asarray(atr_pct_samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("atr_pct_samples is empty or non-finite; cannot compute prod-aligned TP params")

    h_list = [int(h) for h in horizons]
    if worst_horizon not in h_list:
        h_list = sorted(set(h_list + [int(worst_horizon)]))

    scales = {h: float(horizon_scaling_fn(int(h))) for h in h_list}
    s2 = float(scales.get(2, horizon_scaling_fn(2)))
    s4 = float(scales.get(4, horizon_scaling_fn(4)))
    s8 = float(scales.get(8, horizon_scaling_fn(8)))

    quantile_vals = {float(qv): float(np.quantile(arr, float(np.clip(qv, 0.0, 1.0)))) for qv in q_grid}

    tp_min_tradeable = float(max(float(margin_mult) * float(fee_pct_total), float(hard_min_tp)))
    s_worst = float(scales.get(int(worst_horizon), 1.0))
    s_worst_safe = max(abs(s_worst), 1e-12)

    candidates: List[Dict[str, Any]] = []
    for qv in q_grid:
        qvf = float(qv)
        atr_q = float(quantile_vals[qvf])
        for a in alpha_grid:
            alpha_i = float(a)
            tp_center_h2 = alpha_i * atr_q
            tp_eff_h2 = float(np.clip(tp_center_h2, float(h2_lower), float(h2_upper)))
            tp_base_pct = tp_eff_h2 / max(abs(s2), 1e-12)
            tp_base_pct = max(tp_base_pct, tp_min_tradeable / s_worst_safe)
            tp_base_pct *= float(inflate)
            tp_eff_h4 = tp_base_pct * s4
            tp_eff_h8 = tp_base_pct * s8
            candidates.append(
                {
                    "tp_base_pct": float(tp_base_pct),
                    "tp_eff_targets": {"H2": float(tp_eff_h2), "H4": float(tp_eff_h4), "H8": float(tp_eff_h8)},
                    "tp_eff_bands": {
                        "H2": [float(h2_lower), float(h2_upper)],
                        "H4": [float(h2_lower) * (s4 / max(abs(s2), 1e-12)), float(h2_upper) * (s4 / max(abs(s2), 1e-12))],
                        "H8": [float(h2_lower) * (s8 / max(abs(s2), 1e-12)), float(h2_upper) * (s8 / max(abs(s2), 1e-12))],
                    },
                    "atr_q": float(atr_q),
                    "q": qvf,
                    "alpha": alpha_i,
                    "scaling": {"s2": float(s2), "s4": float(s4), "s8": float(s8)},
                    "tp_center_h2": float(tp_center_h2),
                }
            )

    # stable dedup by rounded tp_base
    uniq: Dict[float, Dict[str, Any]] = {}
    for c in candidates:
        k = round(float(c["tp_base_pct"]), 6)
        if k not in uniq:
            uniq[k] = c
    candidates_u = sorted(uniq.values(), key=lambda x: float(x["tp_base_pct"]))

    qv = float(np.clip(q, 0.0, 1.0))
    atr_q = float(np.quantile(arr, qv))
    tp_atr_anchor = float(alpha) * atr_q
    tp_base = max(tp_atr_anchor, tp_min_tradeable / s_worst_safe)
    tp_base *= float(inflate)

    return {
        "tp_base_pct": float(tp_base),
        "tp_abs_lo_pct": float(tp_min_tradeable),
        "tp_min_tradeable": float(tp_min_tradeable),
        "atr_q": float(atr_q),
        "q": float(qv),
        "alpha": float(alpha),
        "margin_mult": float(margin_mult),
        "fee_pct_total": float(fee_pct_total),
        "inflate": float(inflate),
        "worst_horizon": int(worst_horizon),
        "s_worst": float(s_worst),
        "tp_atr_anchor": float(tp_atr_anchor),
        "tp_base_candidates": candidates_u,
        "h2_band": [float(h2_lower), float(h2_upper)],
        "horizons": h_list,
        "scaling": {"s2": float(s2), "s4": float(s4), "s8": float(s8)},
        "atr_quantiles": {"q50": quantile_vals.get(0.5, float('nan')), "q75": quantile_vals.get(0.75, float('nan')), "q90": quantile_vals.get(0.9, float('nan'))},
    }


def production_admissibility_report(
    *,
    events_prod: pd.DataFrame,
    score_prod: np.ndarray,
    bucket_horizon_metrics_prod: Dict[str, Dict[str, Any]],
    tp_lo_prod: float,
    sl_lo_prod: Optional[float] = None,
    gates: ProdGates = ProdGates(),
    col_bucket: str = "bucket",
    col_h: str = "horizon",
    col_label: str = "label",
    col_tp_eff: str = "tp",
    col_sl_eff: str = "sl",
    col_payoff: str = "payoff",
    auc_consistency_tol: float = 0.03,
) -> Dict[str, Any]:
    failures = []
    out: Dict[str, Any] = {"admissible_tier0": False, "failures": failures, "per_cell_health": {}, "aggregates": {}}

    if events_prod is None or events_prod.empty:
        failures.append("U_prod is empty: no production-evaluable events after candidate/RR filters.")
        return out
    if len(score_prod) != len(events_prod):
        failures.append(f"score_prod length mismatch: score={len(score_prod)} events={len(events_prod)}")
        return out

    y = events_prod[col_label].to_numpy()
    s = np.asarray(score_prod, dtype=np.float64)
    tp_hit = float(np.mean(y == 1))
    sl_hit = float(np.mean(y == -1))
    timeout = float(np.mean(y == 0))
    bind = tp_hit + sl_hit
    sl_to_tp = sl_hit / max(tp_hit, EPS)

    tp_eff = events_prod[col_tp_eff].to_numpy(dtype=np.float64, copy=False)
    tp_floor_bind_agg = compute_floor_binding(tp_eff, tp_lo_prod)

    sl_floor_bind_agg = float("nan")
    if sl_lo_prod is not None and col_sl_eff in events_prod.columns:
        sl_eff = events_prod[col_sl_eff].to_numpy(dtype=np.float64, copy=False)
        sl_floor_bind_agg = compute_floor_binding(sl_eff, float(sl_lo_prod))

    out["aggregates"] = {
        "n_prod": int(len(events_prod)),
        "tp_hit_prod": tp_hit,
        "sl_hit_prod": sl_hit,
        "timeout_prod": timeout,
        "bind_prod": bind,
        "sl_to_tp_prod": sl_to_tp,
        "tp_floor_bind_prod_agg": tp_floor_bind_agg,
        "sl_floor_bind_prod_agg": sl_floor_bind_agg,
        "tp_lo_prod": float(tp_lo_prod),
        "sl_lo_prod": float(sl_lo_prod) if sl_lo_prod is not None else None,
    }

    min_n = float("inf")
    min_bind_cell = float("inf")
    max_sl_to_tp_cell = 0.0
    min_auc = float("inf")
    min_auc_bound = float("inf")
    min_sep = float("inf")
    min_ap_lift = float("inf")
    min_tp_over_sl = float("inf")
    max_tp_floor_bind_cell = 0.0
    max_sl_floor_bind_cell = 0.0
    timeout_by_cell: Dict[str, float] = {ck: float("nan") for ck in CANON_CELLS}

    for (b, h), g in events_prod.groupby([col_bucket, col_h], observed=True):
        ck = _cell_key(str(b), int(h))
        if ck not in CANON_CELLS:
            continue
        n_cell = int(len(g))
        min_n = min(min_n, n_cell)

        y_cell = g[col_label].to_numpy()
        idx = g.index.to_numpy(dtype=np.int64, copy=False)
        s_cell = s[idx]

        tp_cell = float(np.mean(y_cell == 1))
        sl_cell = float(np.mean(y_cell == -1))
        to_cell = float(np.mean(y_cell == 0))
        timeout_by_cell[ck] = to_cell
        bind_cell = tp_cell + sl_cell
        min_bind_cell = min(min_bind_cell, bind_cell)

        sl_to_tp_cell = sl_cell / max(tp_cell, EPS)
        max_sl_to_tp_cell = max(max_sl_to_tp_cell, sl_to_tp_cell)

        tp_fb = compute_floor_binding(g[col_tp_eff].to_numpy(dtype=np.float64, copy=False), tp_lo_prod)
        if not math.isfinite(tp_fb):
            failures.append(f"{ck} tp_floor_bind unavailable (non-finite TP).")
        else:
            max_tp_floor_bind_cell = max(max_tp_floor_bind_cell, tp_fb)

        sl_fb = float("nan")
        if sl_lo_prod is not None and col_sl_eff in g.columns:
            sl_fb = compute_floor_binding(g[col_sl_eff].to_numpy(dtype=np.float64, copy=False), float(sl_lo_prod))
            if not math.isfinite(sl_fb):
                failures.append(f"{ck} sl_floor_bind unavailable (non-finite SL).")
            else:
                max_sl_floor_bind_cell = max(max_sl_floor_bind_cell, sl_fb)

        m = bucket_horizon_metrics_prod.get(ck, {})
        auc = _safe_float(m.get("auc_label"), float("nan"))
        auc_b = _safe_float(m.get("auc_bound"), float("nan"))
        sep = _safe_float(m.get("tp_sep_top10"), float("nan"))
        ap_lift = _safe_float(m.get("ap_lift"), float("nan"))
        tp_over_sl_metric = _safe_float(m.get("tp_over_sl"), float("nan"))

        # Recompute per-cell AUC from score_prod for consistency checks.
        y_tp = (y_cell == 1).astype(np.int8)
        n_pos = int(y_tp.sum())
        n_neg = len(y_tp) - n_pos
        auc_from_score = float("nan")
        if n_pos > 0 and n_neg > 0:
            ranks = pd.Series(s_cell).rank(method="average").to_numpy(np.float64)
            u = ranks[y_tp == 1].sum() - n_pos * (n_pos + 1) / 2.0
            auc_from_score = float(u / (n_pos * n_neg))
            if math.isfinite(auc) and abs(auc - auc_from_score) > float(auc_consistency_tol):
                failures.append(
                    f"{ck} auc_label mismatch vs score_prod (provided={auc:.4f}, recomputed={auc_from_score:.4f}, tol={auc_consistency_tol:.4f})"
                )

        if math.isfinite(auc):
            min_auc = min(min_auc, auc)
        if math.isfinite(auc_b):
            min_auc_bound = min(min_auc_bound, auc_b)
        if math.isfinite(sep):
            min_sep = min(min_sep, sep)
        if math.isfinite(ap_lift):
            min_ap_lift = min(min_ap_lift, ap_lift)
        if math.isfinite(tp_over_sl_metric):
            min_tp_over_sl = min(min_tp_over_sl, tp_over_sl_metric)

        out["per_cell_health"][ck] = {
            "n_prod": n_cell,
            "tp_hit_prod": tp_cell,
            "sl_hit_prod": sl_cell,
            "timeout_prod": to_cell,
            "bind_prod": bind_cell,
            "sl_to_tp_prod": sl_to_tp_cell,
            "tp_floor_bind_prod": tp_fb,
            "sl_floor_bind_prod": sl_fb,
            "auc_prod": auc,
            "auc_bound_prod": auc_b,
            "tp_sep_prod": sep,
            "ap_lift_prod": ap_lift,
            "tp_over_sl_prod": tp_over_sl_metric,
            "auc_from_score_prod": auc_from_score,
        }

    covered = set(out["per_cell_health"].keys())
    missing = [ck for ck in CANON_CELLS if ck not in covered]
    if missing:
        failures.append(f"Missing canonical cells in U_prod: {missing}")

    if min_n == float("inf"):
        failures.append("No per-cell groups found (bucket,horizon) in U_prod.")
        return out

    timeout_vals = [timeout_by_cell[ck] for ck in CANON_CELLS if math.isfinite(timeout_by_cell.get(ck, float("nan")))]
    timeout_range = float(np.max(timeout_vals) - np.min(timeout_vals)) if len(timeout_vals) >= 2 else float("nan")
    out["aggregates"]["timeout_range_prod"] = timeout_range

    if min_n < gates.n_min:
        failures.append(f"min_cell_n_prod {min_n} < n_min {gates.n_min}")
    if min_bind_cell < gates.bind_cell_min:
        failures.append(f"min_cell_bind_prod {min_bind_cell:.3f} < bind_cell_min {gates.bind_cell_min:.3f}")
    if bind < gates.bind_min:
        failures.append(f"bind_prod_agg {bind:.3f} < bind_min {gates.bind_min:.3f}")
    if gates.tp_hit_min_agg > 0.0 and tp_hit < gates.tp_hit_min_agg:
        failures.append(f"tp_hit_prod_agg {tp_hit:.3f} < tp_hit_min_agg {gates.tp_hit_min_agg:.3f}")
    if gates.enforce_tradeable_tp_lo and float(tp_lo_prod) < float(gates.tradeable_tp_min):
        failures.append(f"tp_lo_prod {float(tp_lo_prod):.4f} < tradeable_tp_min {float(gates.tradeable_tp_min):.4f}")
    if timeout > gates.timeout_max:
        failures.append(f"timeout_prod_agg {timeout:.3f} > timeout_max {gates.timeout_max:.3f}")
    if math.isfinite(timeout_range) and timeout_range > gates.timeout_range_max:
        failures.append(f"timeout_range_prod {timeout_range:.3f} > timeout_range_max {gates.timeout_range_max:.3f}")
    if sl_to_tp > gates.sl_to_tp_max:
        failures.append(f"sl_to_tp_prod_agg {sl_to_tp:.2f}x > sl_to_tp_max {gates.sl_to_tp_max:.2f}x")
    if max_sl_to_tp_cell > gates.sl_to_tp_max:
        failures.append(f"max_cell_sl_to_tp_prod {max_sl_to_tp_cell:.2f}x > sl_to_tp_max {gates.sl_to_tp_max:.2f}x")

    if min_tp_over_sl != float("inf") and min_tp_over_sl < gates.tp_over_sl_min:
        failures.append(f"min_cell_tp_over_sl {min_tp_over_sl:.3f} < tp_over_sl_min {gates.tp_over_sl_min:.3f}")
    elif min_tp_over_sl == float("inf"):
        failures.append("min_cell_tp_over_sl unavailable (missing tp_over_sl in per-cell metrics).")

    if min_auc != float("inf") and min_auc < gates.auc_min:
        failures.append(f"min_cell_auc {min_auc:.4f} < auc_min {gates.auc_min:.4f}")
    elif min_auc == float("inf"):
        failures.append("min_cell_auc unavailable (missing auc_label in per-cell metrics).")

    if min_auc_bound != float("inf") and min_auc_bound < gates.auc_bound_min:
        failures.append(f"min_cell_auc_bound {min_auc_bound:.4f} < auc_bound_min {gates.auc_bound_min:.4f}")
    elif min_auc_bound == float("inf"):
        failures.append("min_cell_auc_bound unavailable (missing auc_bound in per-cell metrics).")

    if min_sep != float("inf") and min_sep < gates.tp_sep_min:
        failures.append(f"min_cell_tp_sep {min_sep:.4f} < tp_sep_min {gates.tp_sep_min:.4f}")
    elif min_sep == float("inf"):
        failures.append("min_cell_tp_sep unavailable (missing tp_sep_top10 in per-cell metrics).")

    if min_ap_lift != float("inf") and min_ap_lift < gates.ap_lift_min:
        failures.append(f"min_cell_ap_lift {min_ap_lift:.3f} < ap_lift_min {gates.ap_lift_min:.3f}")
    elif min_ap_lift == float("inf"):
        failures.append("min_cell_ap_lift unavailable (missing ap_lift in per-cell metrics).")

    if math.isfinite(tp_floor_bind_agg) and tp_floor_bind_agg > gates.tp_floor_bind_max_agg:
        failures.append(
            f"tp_floor_bind_prod_agg {tp_floor_bind_agg*100:.1f}% > tp_floor_bind_max_agg {gates.tp_floor_bind_max_agg*100:.1f}%"
        )
    if max_tp_floor_bind_cell > gates.tp_floor_bind_max_cell:
        failures.append(
            f"max_cell_tp_floor_bind_prod {max_tp_floor_bind_cell*100:.1f}% > tp_floor_bind_max_cell {gates.tp_floor_bind_max_cell*100:.1f}%"
        )

    if gates.sl_floor_bind_max_cell is not None and sl_lo_prod is not None:
        if max_sl_floor_bind_cell > float(gates.sl_floor_bind_max_cell):
            failures.append(
                f"max_cell_sl_floor_bind_prod {max_sl_floor_bind_cell*100:.1f}% > sl_floor_bind_max_cell {float(gates.sl_floor_bind_max_cell)*100:.1f}%"
            )

    if (math.isfinite(tp_floor_bind_agg) and tp_floor_bind_agg > 0.50) or (max_tp_floor_bind_cell > 0.60):
        req_auc = gates.auc_min + float(gates.floor_bind_auc_boost)
        if min_auc == float("inf") or min_auc < req_auc:
            failures.append(
                f"floor-dominance combined guardrail: min_cell_auc {min_auc if min_auc!=float('inf') else float('nan'):.4f} < required {req_auc:.4f} when TP floor-binding is high"
            )

    out["aggregates"].update(
        {
            "min_cell_n_prod": int(min_n),
            "min_cell_bind_prod": float(min_bind_cell),
            "max_cell_sl_to_tp_prod": float(max_sl_to_tp_cell),
            "min_cell_auc_prod": float(min_auc) if min_auc != float("inf") else float("nan"),
            "min_cell_auc_bound_prod": float(min_auc_bound) if min_auc_bound != float("inf") else float("nan"),
            "min_cell_tp_sep_prod": float(min_sep) if min_sep != float("inf") else float("nan"),
            "min_cell_ap_lift_prod": float(min_ap_lift) if min_ap_lift != float("inf") else float("nan"),
            "min_cell_tp_over_sl_prod": float(min_tp_over_sl) if min_tp_over_sl != float("inf") else float("nan"),
            "max_cell_tp_floor_bind_prod": float(max_tp_floor_bind_cell),
            "max_cell_sl_floor_bind_prod": float(max_sl_floor_bind_cell) if sl_lo_prod is not None else None,
            "auc_consistency_tol": float(auc_consistency_tol),
            "tradeable_tp_min": float(gates.tradeable_tp_min),
        }
    )

    out["admissible_tier0"] = len(failures) == 0
    return out


def econ_guardrail_factor(
    tp_hit_agg: float,
    sl_to_tp_agg: float,
    tp_over_sl: float,
    *,
    tp_hit_floor: float = 0.10,
    sl_to_tp_cap: float = 2.5,
    tp_over_sl_floor: float = 1.05,
) -> float:
    """Returns an economic guardrail factor in [0, 1] using weak-link geometric mean."""
    g_tp = np.clip(tp_hit_agg / max(tp_hit_floor, EPS), 0.0, 1.0)
    g_bal = np.clip(sl_to_tp_cap / max(sl_to_tp_agg, EPS), 0.0, 1.0)
    g_edge = np.clip(tp_over_sl / max(tp_over_sl_floor, EPS), 0.0, 1.0)
    return float((g_tp * g_bal * g_edge) ** (1.0 / 3.0))


def econ_admissible(
    tp_hit_agg: float,
    sl_to_tp_agg: float,
    tp_over_sl: float,
    *,
    tp_hit_floor: float = 0.10,
    sl_to_tp_cap: float = 2.5,
    tp_over_sl_floor: float = 1.05,
    min_factor: float = 0.85,
) -> bool:
    """Hard economic gate on each leg and composite guardrail factor."""
    if tp_hit_agg < tp_hit_floor:
        return False
    if sl_to_tp_agg > sl_to_tp_cap:
        return False
    if tp_over_sl < tp_over_sl_floor:
        return False
    return (
        econ_guardrail_factor(
            tp_hit_agg,
            sl_to_tp_agg,
            tp_over_sl,
            tp_hit_floor=tp_hit_floor,
            sl_to_tp_cap=sl_to_tp_cap,
            tp_over_sl_floor=tp_over_sl_floor,
        )
        >= min_factor
    )


def apply_econ_guardrail_to_stage2(
    stage2_score: float,
    tp_hit_agg: float,
    sl_to_tp_agg: float,
    tp_over_sl: float,
    *,
    min_factor: float = 0.85,
    mult_floor: float = 0.70,
    mult_weight: float = 0.30,
    add_bonus_max: float = 0.05,
    tp_hit_floor: float = 0.10,
    sl_to_tp_cap: float = 2.5,
    tp_over_sl_floor: float = 1.05,
) -> tuple[float, bool, float, float]:
    """Apply economic guardrail to stage2 score.

    Returns (new_stage2_score, econ_ok, econ_G, econ_multiplier).
    """
    G = econ_guardrail_factor(
        tp_hit_agg,
        sl_to_tp_agg,
        tp_over_sl,
        tp_hit_floor=tp_hit_floor,
        sl_to_tp_cap=sl_to_tp_cap,
        tp_over_sl_floor=tp_over_sl_floor,
    )
    ok = econ_admissible(
        tp_hit_agg,
        sl_to_tp_agg,
        tp_over_sl,
        tp_hit_floor=tp_hit_floor,
        sl_to_tp_cap=sl_to_tp_cap,
        tp_over_sl_floor=tp_over_sl_floor,
        min_factor=min_factor,
    )
    mult = float(np.clip(mult_floor + mult_weight * G, 0.0, 10.0))
    out = stage2_score * mult
    if add_bonus_max > 0.0 and G > min_factor and min_factor < 1.0:
        bonus = add_bonus_max * float(np.clip((G - min_factor) / (1.0 - min_factor), 0.0, 1.0))
        out += bonus
    return out, ok, G, mult
