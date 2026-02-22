from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import math

EPS = 1e-12


@dataclass(frozen=True)
class SLTPPolicy:
    """
    Production-aligned widening of SL/TP exploration while enforcing:
        TP_eff >= SL_eff + superiority_add

    Where superiority_add is in absolute pct units.
    """

    sl_as_tp_pct_grid: Tuple[float, ...] = (
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
        1.00,
        1.25,
    )
    superiority_add: float = 0.075
    drop_on_violation: bool = True


def _isfinite_pos(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def passes_tp_superior_additive(tp_eff: float, sl_eff: float, superiority_add: float) -> bool:
    """Enforce: tp_eff >= sl_eff + superiority_add."""
    if not (_isfinite_pos(tp_eff) and _isfinite_pos(sl_eff) and math.isfinite(superiority_add)):
        return False
    return (tp_eff + EPS) >= (sl_eff + max(float(superiority_add), 0.0))


def expand_configs_wide_sl_tp_additive_superiority(
    base_cfg: Dict[str, Any],
    *,
    tp_eff: float,
    policy: SLTPPolicy = SLTPPolicy(),
) -> List[Dict[str, Any]]:
    """Expand a base cfg across sl_as_tp_pct ladder with additive TP superiority gating."""
    out: List[Dict[str, Any]] = []

    if not _isfinite_pos(tp_eff):
        cfg = dict(base_cfg)
        cfg.setdefault("prod_aligned_tp", {})
        cfg["prod_aligned_tp"]["tp_superiority_rule"] = {
            "mode": "additive",
            "superiority_add": float(policy.superiority_add),
            "passes": False,
            "reason": f"invalid tp_eff={tp_eff}",
        }
        return [] if policy.drop_on_violation else [cfg]

    for s in policy.sl_as_tp_pct_grid:
        cfg = dict(base_cfg)
        cfg["sl_method"] = "tp_pct"
        cfg["sl_as_tp_pct"] = float(s)

        sl_eff = float(s) * float(tp_eff)
        ok = passes_tp_superior_additive(tp_eff, sl_eff, policy.superiority_add)

        cfg.setdefault("prod_aligned_tp", {})
        cfg["prod_aligned_tp"].update(
            {
                "sl_as_tp_pct_candidate": float(s),
                "tp_eff_used": float(tp_eff),
                "sl_eff_implied": float(sl_eff),
                "tp_superiority_rule": {
                    "mode": "additive",
                    "superiority_add": float(policy.superiority_add),
                    "lhs_tp_eff": float(tp_eff),
                    "rhs_sl_plus_add": float(sl_eff + policy.superiority_add),
                    "passes": bool(ok),
                },
            }
        )

        if ok or not policy.drop_on_violation:
            out.append(cfg)

    return out
